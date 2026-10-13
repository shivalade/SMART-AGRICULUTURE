from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# `url_parse` moved in some Werkzeug versions; provide a compatible fallback to
# `urllib.parse.urlparse` so older import sites continue to work across envs.
try:
    from werkzeug.urls import url_parse
except Exception:
    from urllib.parse import urlparse as url_parse
import random
import os
import json
import time
import logging
import uuid
from datetime import datetime, timedelta
import warnings
import io
from PIL import Image
HAS_MPL = True
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except Exception as e:
    HAS_MPL = False
    plt = None
    np = None
    print(f"Warning: matplotlib/numpy not available: {e}")

# numpy compatibility shim: `np.alltrue` was removed in NumPy 2.0
# Some third-party libs (NetworkX/Matplotlib) or older code may still call
# `np.alltrue`. Provide a backward-compatible alias to `np.all` when numpy
# is available so runtime calls don't fail.
if np is not None:
    try:
        if not hasattr(np, 'alltrue'):
            np.alltrue = np.all
    except Exception:
        # Non-fatal: if assignment fails, continue without shim
        pass

# Import NetworkX after NumPy shim so legacy calls like `np.alltrue` in older
# releases of NetworkX don't break when NumPy 2.0 removed `alltrue`.
try:
    import networkx as nx
except Exception as e:
    nx = None
    print(f"Warning: networkx import failed: {e}")

# Suppress SQLAlchemy LegacyAPIWarning
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*legacy.*')

# ============================
# Database Setup
# ============================
app = Flask(__name__, template_folder='.')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///wsn_routing.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.environ.get('FLASK_SECRET_KEY') or os.urandom(32)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ============================
# Database Models
# ============================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    calculations = db.relationship('Calculation', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password, password)


class Calculation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    source = db.Column(db.Integer, nullable=False)
    destination = db.Column(db.Integer, nullable=False)

    # Flooding metrics
    flooding_hops = db.Column(db.Integer)
    flooding_delay = db.Column(db.Float)
    flooding_throughput = db.Column(db.Float)
    flooding_exec_time = db.Column(db.Float)
    flooding_energy = db.Column(db.Float)
    flooding_packet_loss = db.Column(db.Float)
    
    # LoRaWAN metrics
    lorawan_hops = db.Column(db.Integer)
    lorawan_delay = db.Column(db.Float)
    lorawan_throughput = db.Column(db.Float)
    lorawan_exec_time = db.Column(db.Float)
    lorawan_energy = db.Column(db.Float)
    lorawan_packet_loss = db.Column(db.Float)
    
    # Legacy columns (kept for compatibility)
    packet_loss = db.Column(db.Float)
    energy = db.Column(db.Float)
    num_nodes = db.Column(db.Integer, default=25)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to paths
    paths = db.relationship('RoutingPath', backref='calculation', lazy=True, cascade='all, delete-orphan')


class RoutingPath(db.Model):
    """Store routing paths for each algorithm"""
    id = db.Column(db.Integer, primary_key=True)
    calculation_id = db.Column(db.Integer, db.ForeignKey('calculation.id'), nullable=False)
    algorithm = db.Column(db.String(50), nullable=False)  # Flooding, LoRaWAN, Energy-Aware
    path_data = db.Column(db.Text, nullable=False)  # JSON string of path nodes
    path_length = db.Column(db.Integer, nullable=False)  # Number of hops
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def get_path(self):
        """Convert JSON path data back to list"""
        return json.loads(self.path_data)
    
    def set_path(self, path_list):
        """Store path as JSON"""
        self.path_data = json.dumps(path_list)
        self.path_length = len(path_list)


class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(50))


# ============================
# Advanced Features Models
# ============================
class NetworkSimulation(db.Model):
    """Store network simulation results with node failures"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    num_nodes = db.Column(db.Integer, default=25)
    failed_nodes = db.Column(db.Text)  # JSON array of failed node IDs
    simulation_data = db.Column(db.Text)  # JSON results
    success_rate = db.Column(db.Float)  # Percentage of successful routes
    avg_delay = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class CustomTopology(db.Model):
    """Store custom network topologies created by users"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    topology_json = db.Column(db.Text)  # JSON representation of graph
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class PerformanceMetrics(db.Model):
    """Advanced performance analytics"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    calculation_id = db.Column(db.Integer, db.ForeignKey('calculation.id'))
    algorithm = db.Column(db.String(50))
    energy_consumed = db.Column(db.Float)
    packet_delivery_ratio = db.Column(db.Float)
    end_to_end_delay = db.Column(db.Float)
    jitter = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ============================
# Logging Configuration
# ============================
logging.basicConfig(
    filename='wsn_routing.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_audit(user_id, action, details):
    """Log audit trail to database"""
    audit = AuditLog(
        user_id=user_id,
        action=action,
        details=details,
        ip_address=request.remote_addr
    )
    db.session.add(audit)
    db.session.commit()
    logger.info(f"User {user_id}: {action} - {details}")



# ============================
# Create Network
# ============================
def create_network(num_nodes=25, radius=0.4):
    G = nx.random_geometric_graph(num_nodes, radius)
    for u, v in G.edges():
        G[u][v]['weight'] = round(random.uniform(1, 10), 2)
    return G


# ============================
# Create Network from JSON
# ============================
def create_network_from_json(json_data):
    """Create network from uploaded JSON topology"""
    try:
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
        G = nx.Graph()
        
        # Add nodes
        for node in data.get('nodes', []):
            G.add_node(node)
        
        # Add edges with weights
        for edge in data.get('edges', []):
            src = edge['source']
            dst = edge['target']
            weight = edge.get('weight', 1.0)
            G.add_edge(src, dst, weight=weight)
        
        return G
    except Exception as e:
        logger.error(f"Error parsing topology JSON: {e}")
        return None


# ============================
# Routing Algorithms with Timing
# ============================



# ============================
# Advanced Routing Algorithms
# ============================
def energy_aware_path(G, src, dst):
    """Find path minimizing energy consumption"""
    try:
        if src not in G.nodes() or dst not in G.nodes():
            return None, 0
        start_time = time.time()
        # Weight by energy consumption (hop count * energy per hop)
        for u, v in G.edges():
            G[u][v]['energy'] = G[u][v].get('weight', 1) * random.uniform(0.8, 1.2)
        path = nx.dijkstra_path(G, src, dst, weight='energy')
        exec_time = (time.time() - start_time) * 1000
        return path, exec_time
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, 0


def multi_constrained_path(G, src, dst, constraints={'delay': 0.4, 'energy': 0.4, 'hops': 0.2}):
    """Find path optimizing multiple constraints (weighted multi-objective)"""
    try:
        if src not in G.nodes() or dst not in G.nodes():
            return None, 0
        start_time = time.time()
        
        # Try to find all simple paths
        try:
            paths = list(nx.all_simple_paths(G, src, dst, cutoff=5))
        except nx.NetworkXNoPath:
            return None, time.time() - start_time
        
        if not paths:
            return None, time.time() - start_time
        
        # Score each path
        best_path = None
        best_score = float('inf')
        
        for path in paths:
            hops = len(path) - 1
            delay = sum(G[path[i]][path[i+1]].get('weight', 1) for i in range(hops)) if hops > 0 else 0
            energy = hops * random.uniform(0.8, 1.2)
            
            # Weighted score
            score = (constraints.get('delay', 0.4) * delay +
                    constraints.get('energy', 0.4) * energy +
                    constraints.get('hops', 0.2) * hops)
            
            if score < best_score:
                best_score = score
                best_path = path
        
        exec_time = (time.time() - start_time) * 1000
        return best_path, exec_time
    except Exception:
        return None, 0


def simulate_network_failure(G, failed_nodes):
    """Simulate network with failed nodes"""
    G_copy = G.copy()
    for node in failed_nodes:
        if node in G_copy.nodes():
            G_copy.remove_node(node)
    return G_copy


def flooding_path(G, src, dst):
    """Return shortest path and execution time for flooding (timed)."""
    try:
        if src not in G.nodes() or dst not in G.nodes():
            return None, 0
        start_time = time.time()
        path = nx.shortest_path(G, src, dst)
        exec_time = (time.time() - start_time) * 1000
        return path, exec_time
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, 0


# ============================
# Compute Metrics
# ============================
def compute_metrics(G, path, algo=None):
    """
    Compute comprehensive routing metrics for LoRaWAN vs Flooding protocols
    """
    if path is None:
        return None

    hops = len(path) - 1
    
    # Calculate base delay from edge weights
    edge_weights = []
    for i in range(hops):
        if G.has_edge(path[i], path[i+1]):
            weight = G[path[i]][path[i+1]].get('weight', 1)
            # Ensure weight is a valid number
            if weight is None or weight == 0:
                weight = 1
            edge_weights.append(weight)
        else:
            edge_weights.append(1)
    
    base_delay = sum(edge_weights) if edge_weights else 0
    
    # Ensure minimum delay - even 1-hop direct connections have some latency
    if base_delay <= 0:
        base_delay = max(10, hops * 8)  # 8ms per hop, minimum 10ms
    else:
        base_delay = max(10, base_delay)  # Ensure at least 10ms
    
    # Ensure minimum throughput
    base_throughput = max(50, 1024 / base_delay)  # At least 50 kbps
    
    # Calculate packet loss based on hops
    base_packet_loss = max(0.5, hops * 1.0)  # 0.5% minimum, 1% per hop

    # Energy calculation: sum of edge energy or use weight as proxy
    energy_vals = []
    for i in range(hops):
        if G.has_edge(path[i], path[i+1]):
            edge = G.get_edge_data(path[i], path[i+1], default={})
            energy = edge.get('energy', edge.get('weight', 2))
            if energy is None or energy <= 0:
                energy = 2
            energy_vals.append(energy)
        else:
            energy_vals.append(2)

    if energy_vals:
        base_energy = sum(energy_vals)
    else:
        base_energy = max(4, hops * 4)  # 4mJ per hop baseline, minimum 4mJ

    # Compute metrics per algorithm. 'hops' returned is the raw hop count (len(path)-1).
    if algo and algo.lower() == 'lorawan':
        energy = round(base_energy * 0.35, 2)
        delay = round(max(8, base_delay * 0.55), 2)
        throughput = round(max(base_throughput * 1.4, 80), 2)
        packet_loss = round(max(0.2, base_packet_loss * 0.25), 2)
    else:
        # Flooding or default
        energy = round(base_energy * 1.2, 2)
        delay = round(base_delay * 1.3, 2)
        throughput = round(max(base_throughput * 0.75, 40), 2)
        packet_loss = round(base_packet_loss * 1.5, 2)

    result_dict = {
        'hops': hops,
        'delay': delay,
        'throughput': throughput,
        'packet_loss': packet_loss,
        'energy': energy
    }
    print(f"DEBUG compute_metrics: algo={algo}, path_len={len(path)}, result={result_dict}")
    return result_dict


# ============================
# Draw Graph with Paths
# ============================
def draw_individual_graphs(G, paths, source=None, dest=None, image_id=''):
    """Draw separate graphs for each algorithm"""
    if not HAS_MPL:
        print("Plotting disabled: matplotlib/numpy not available")
        return False

    try:
        # Ensure static directory exists
        os.makedirs('static', exist_ok=True)
        
        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            pos = nx.spring_layout(G, seed=42)
        
        # Color mapping for algorithms
        colors = {
            'Flooding': '#ef4444',
            'LoRaWAN': '#059669'
        }
        
        # Generate individual graphs for each algorithm
        for algo_name, path in paths.items():
            plt.figure(figsize=(8, 8))
            
            # Draw base network with light gray
            nx.draw_networkx_nodes(G, pos, node_color='#f3f4f6', node_size=500, 
                                  edgecolors='#9ca3af', linewidths=1.5)
            nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', 
                                   font_color='#1f2937')
            nx.draw_networkx_edges(G, pos, width=1, alpha=0.2, edge_color='#d1d5db')
            
            # Highlight the algorithm's path
            if path and len(path) > 1:
                edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                nx.draw_networkx_edges(G, pos, edgelist=edges, 
                                      edge_color=colors[algo_name], 
                                      width=4, alpha=0.9)
                
                # Highlight path nodes
                nx.draw_networkx_nodes(G, pos, nodelist=path,
                                      node_color=colors[algo_name], node_size=600,
                                      alpha=0.8)
            
            # Highlight source and destination
            source_node, dest_node = path[0], path[-1]
            nx.draw_networkx_nodes(G, pos, nodelist=[source_node], 
                                  node_color='#10b981', node_size=900,
                                  edgecolors='#047857', linewidths=2.5)
            nx.draw_networkx_nodes(G, pos, nodelist=[dest_node], 
                                  node_color='#ef4444', node_size=900,
                                  edgecolors='#991b1b', linewidths=2.5)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#10b981', edgecolor='#047857', label='Source Node'),
                Patch(facecolor='#ef4444', edgecolor='#991b1b', label='Destination Node'),
                Patch(facecolor=colors[algo_name], label=f'{algo_name} Path')
            ]
            plt.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95)
            
            plt.title(f"{algo_name} Algorithm - Shortest Path Route\nPath: {' → '.join(map(str, path))}", 
                     fontsize=13, fontweight='bold', pad=20)
            plt.axis('off')
            plt.tight_layout()
            
            # Save with algorithm-specific filename (with optional image_id for uniqueness)
            algo_lower = algo_name.lower()
            filename = f"static/graph_{algo_lower}_{image_id}.png" if image_id else f"static/graph_{algo_lower}.png"
            plt.savefig(filename, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close()
        
        return True
    except Exception as e:
        print(f"Error drawing individual graphs: {e}")
        logger.error(f"Error drawing individual graphs: {e}")
        if HAS_MPL and plt:
            plt.close()
        return False


def draw_graph(G, paths, source=None, dest=None, image_id=''):
    """Draw combined graph with all algorithm paths"""
    if not HAS_MPL:
        print("Plotting disabled: matplotlib/numpy not available")
        return False

    try:
        # Ensure static directory exists
        os.makedirs('static', exist_ok=True)
        
        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            pos = nx.spring_layout(G, seed=42)
        
        plt.figure(figsize=(12, 10))
        
        # Draw base network
        nx.draw_networkx_nodes(G, pos, node_color='#f3f4f6', node_size=600,
                              edgecolors='#9ca3af', linewidths=1)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.2, edge_color='#e5e7eb')
        
        # Colors for each algorithm
        colors = {
            'Flooding': '#ef4444',
            'LoRaWAN': '#059669'
        }
        legend_elements = []
        
        # Draw paths with different colors
        for algo_name, path in paths.items():
            if path and len(path) > 1:
                edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                nx.draw_networkx_edges(G, pos, edgelist=edges, 
                                      edge_color=colors[algo_name], 
                                      width=2.5, alpha=0.7)
                legend_elements.append(plt.Line2D([0], [0], color=colors[algo_name], 
                                                  lw=2.5, label=algo_name))
        
        # Highlight source and destination
        if paths and any(p for p in paths.values()):
            first_path = next(p for p in paths.values() if p)
            source_node, dest_node = first_path[0], first_path[-1]
            nx.draw_networkx_nodes(G, pos, nodelist=[source_node], 
                                  node_color='#10b981', node_size=1000,
                                  edgecolors='#047857', linewidths=2)
            nx.draw_networkx_nodes(G, pos, nodelist=[dest_node], 
                                  node_color='#ef4444', node_size=1000,
                                  edgecolors='#991b1b', linewidths=2)
        
        if legend_elements:
            plt.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
        plt.title("Network Routing Topology - All Algorithms Comparison", fontsize=14, fontweight='bold', pad=15)
        plt.axis('off')
        plt.tight_layout()
        
        # Save with optional image_id for uniqueness
        filename = f"static/graph_{image_id}.png" if image_id else "static/graph.png"
        plt.savefig(filename, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Also draw individual graphs for each algorithm
        draw_individual_graphs(G, paths, source, dest, image_id)
        
        return True
    except Exception as e:
        print(f"Error drawing graph: {e}")
        if HAS_MPL and plt:
            plt.close()
        return False


# ============================
# Draw Heatmap for Congestion Analysis
# ============================
def draw_heatmap(G, paths, image_id=''):
    """Draw edge utilization heatmap"""
    if not HAS_MPL:
        print("Plotting disabled: matplotlib/numpy not available")
        return False

    try:
        # Ensure static directory exists
        os.makedirs('static', exist_ok=True)
        
        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            pos = nx.spring_layout(G, seed=42)
        
        # Count edge usage
        edge_usage = {}
        for edge in G.edges():
            edge_usage[edge] = 0
            edge_usage[(edge[1], edge[0])] = 0
        
        # Count path traversals
        for algo_name, path in paths.items():
            if path:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    if (u, v) in edge_usage:
                        edge_usage[(u, v)] += 1
                    if (v, u) in edge_usage:
                        edge_usage[(v, u)] += 1
        
        plt.figure(figsize=(10, 10))
        
        # Draw base network
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=600)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # Color edges based on usage (congestion)
        edge_colors = []
        edge_widths = []
        edges = []
        
        max_usage = max(edge_usage.values()) if edge_usage.values() else 1
        
        for u, v in G.edges():
            usage = max(edge_usage.get((u, v), 0), edge_usage.get((v, u), 0))
            edges.append((u, v))
            
            # Color intensity based on usage
            intensity = usage / max_usage if max_usage > 0 else 0
            if intensity > 0.7:
                edge_colors.append('red')  # High congestion
            elif intensity > 0.4:
                edge_colors.append('orange')  # Medium congestion
            else:
                edge_colors.append('yellow')  # Low congestion
            
            edge_widths.append(1 + intensity * 3)
        
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, 
                              width=edge_widths, alpha=0.8)
        
        plt.title("Network Congestion Heatmap\n(Red=High, Orange=Medium, Yellow=Low)", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save with optional image_id for uniqueness
        filename = f"static/heatmap_{image_id}.png" if image_id else "static/heatmap.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"Error drawing heatmap: {e}")
        if HAS_MPL and plt:
            plt.close()
        return False


app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.permanent_session_lifetime = timedelta(hours=24)
os.makedirs('static', exist_ok=True)


@app.before_request
def make_session_permanent():
    session.permanent = True


@app.after_request
def add_pwa_headers(response):
    """Add PWA and mobile optimization headers"""
    # Cache manifest
    if request.path == '/static/manifest.json':
        response.headers['Cache-Control'] = 'public, max-age=3600'
        response.headers['Content-Type'] = 'application/json'
    
    # Cache service worker with validation
    if request.path == '/static/service-worker.js':
        response.headers['Cache-Control'] = 'public, max-age=3600'
        response.headers['Service-Worker-Allowed'] = '/'
    
    # Disable caching for HTML pages for updates
    if response.content_type and 'text/html' in response.content_type:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    return response


# ============================
# Routes - Authentication
# ============================
@app.route('/')
@login_required
def home():
    """Home page with project statistics"""
    try:
        # Get total calculations for current user
        total_calcs = Calculation.query.filter_by(user_id=current_user.id).count()
        
        # Get total users
        total_users = User.query.count()
        
        # Get all calculations across all users
        all_calcs = Calculation.query.all()
        
        # Calculate average metrics
        if all_calcs:
            # Calculate best average hops from all calculations
            avg_hops_list = []
            for calc in all_calcs:
                for algo in ['flooding', 'lorawan']:
                    hops = getattr(calc, f'{algo}_hops', None)
                    if hops:
                        avg_hops_list.append(hops)
            
            best_avg_hops = round(min(avg_hops_list), 2) if avg_hops_list else 0
        else:
            best_avg_hops = 0
        
        return render_template('homepage.html', 
                             total_calcs=total_calcs,
                             total_users=total_users,
                             best_avg_hops=best_avg_hops,
                             total_calculations=len(all_calcs))
    except Exception as e:
        logger.error(f"Home page error: {e}")
        return render_template('homepage.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            email = request.form.get('email', '').strip()
            password = request.form.get('password', '').strip()
            
            if not username or not email or not password:
                return render_template('register.html', error="All fields required")
            
            if User.query.filter_by(username=username).first():
                return render_template('register.html', error="Username exists")
            
            if User.query.filter_by(email=email).first():
                return render_template('register.html', error="Email exists")
            
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            
            log_audit(user.id, 'USER_REGISTRATION', f'Registered new account: {username}')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            return render_template('register.html', error=str(e))
    
    return render_template('register.html', error=None)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()
            
            user = User.query.filter_by(username=username).first()
            
            if not user or not user.check_password(password):
                return render_template('login.html', error="Invalid credentials")
            
            login_user(user)
            log_audit(user.id, 'USER_LOGIN', f'User logged in: {username}')

            # Respect "next" parameter when present (and ensure it's a safe local URL)
            next_page = request.args.get('next')
            if not next_page or url_parse(next_page).netloc != '':
                return redirect(url_for('home'))
            return redirect(next_page)
        except Exception as e:
            return render_template('login.html', error=str(e))
    
    return render_template('login.html', error=None)


@app.route('/logout')
@login_required
def logout():
    user_id = current_user.id
    username = current_user.username
    logout_user()
    log_audit(user_id, 'USER_LOGOUT', f'User logged out: {username}')
    return redirect(url_for('login'))


# ============================
# Routes - Main Application
# ============================
@app.route('/input', methods=['GET', 'POST'])
@login_required
def input_page():
    if request.method == 'POST':
        try:
            source = request.form.get('source', '').strip()
            destination = request.form.get('destination', '').strip()
            
            if not source or not destination:
                return render_template('input.html', error="Enter both nodes")
            
            try:
                src = int(source)
                dst = int(destination)
            except ValueError:
                return render_template('input.html', error="Invalid numbers")
            
            # Create network
            G = create_network()
            # Validate source/destination against generated network size
            num_nodes = G.number_of_nodes()
            if src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes:
                return render_template('input.html', error=f"Source/Destination must be between 0 and {num_nodes-1}")

            # Define additional simple wrappers for Flooding and LoRaWAN
            def flooding_func(Gf, s, d):
                try:
                    if s not in Gf.nodes() or d not in Gf.nodes():
                        return None, 0
                    start_time = time.time()
                    path = nx.shortest_path(Gf, s, d)
                    exec_time = (time.time() - start_time) * 1000
                    return path, exec_time
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    return None, 0

            # LoRaWAN proposed algorithm - use energy aware routing as representative
            lorawan_func = energy_aware_path

            algorithms = {
                'Flooding': flooding_func,
                'LoRaWAN': lorawan_func
            }
            
            paths = {}
            results = []
            
            for algo_name, algo_func in algorithms.items():
                path, exec_time = algo_func(G, src, dst)
                paths[algo_name] = path
                
                if path:
                    metrics = compute_metrics(G, path, algo_name)
                    if metrics:  # Only add if metrics were calculated
                        results.append({
                            'algorithm': algo_name,
                            'hops': metrics['hops'],
                            'delay': metrics['delay'],
                            'throughput': metrics['throughput'],
                            'packet_loss': metrics['packet_loss'],
                            'energy': metrics['energy'],
                            'exec_time': round(exec_time, 3)
                        })
            
            # Draw visualizations
            draw_graph(G, paths, src, dst)
            draw_heatmap(G, paths)
            
            if not results:
                return render_template('input.html', error="No path found")
            
            # Save to session
            session['graph_results'] = results
            session['graph_source'] = src
            session['graph_destination'] = dst
            session['graph_paths'] = {algo: path for algo, path in paths.items() if path}
            
            # Save to database
            calc = Calculation(
                user_id=current_user.id,
                source=src,
                destination=dst,
                num_nodes=G.number_of_nodes(),
                packet_loss=0,
                energy=0
            )
            
            for result in results:
                algo = result['algorithm'].lower()
                setattr(calc, f'{algo}_hops', result['hops'])
                setattr(calc, f'{algo}_delay', result['delay'])
                setattr(calc, f'{algo}_throughput', result['throughput'])
                setattr(calc, f'{algo}_exec_time', result['exec_time'])
                setattr(calc, f'{algo}_energy', result.get('energy', 0))
                setattr(calc, f'{algo}_packet_loss', result.get('packet_loss', 0))
                print(f"DEBUG: Storing {algo}: delay={result['delay']}, energy={result.get('energy')}, packet_loss={result.get('packet_loss')}")
                
                # Store path information
                if result['algorithm'] in paths and paths[result['algorithm']]:
                    path_obj = RoutingPath(
                        algorithm=result['algorithm'],
                    )
                    path_obj.set_path(paths[result['algorithm']])
                    calc.paths.append(path_obj)
            
            db.session.add(calc)
            db.session.commit()
            
            log_audit(current_user.id, 'ROUTING_CALCULATION', 
                     f'Calculated routes from {src} to {dst}')
            
            # Store calculation data in session for network graphs
            session['calc_num_nodes'] = G.number_of_nodes()
            session['calc_source'] = src
            session['calc_destination'] = dst
            
            return redirect(url_for('network_graphs'))
        except Exception as e:
            return render_template('input.html', error=str(e))
    
    return render_template('input.html', error=None)


@app.route('/graph_view')
@login_required
def graph_view():
    source = session.get('graph_source')
    destination = session.get('graph_destination')
    paths = session.get('graph_paths', {})
    image_id = session.get('graph_image_id', '')
    
    if source is None or destination is None:
        return redirect(url_for('input_page'))
    
    return render_template('graph_display.html', source=source, destination=destination, paths=paths, image_id=image_id)


@app.route('/metrics')
@login_required
def metrics_page():
    # Support viewing metrics for a past calculation via ?calc_id=<id>
    calc_id = request.args.get('calc_id', type=int)
    app.logger.info(f"metrics_page called with calc_id={calc_id}")

    if calc_id:
        calc = Calculation.query.filter_by(id=calc_id, user_id=current_user.id).first()
        if not calc:
            return render_template('metrics.html', error='Calculation not found')

        # Build results structure expected by template
        results = []
        for algo in ['Flooding', 'LoRaWAN']:
            key = algo.lower()
            hops = getattr(calc, f'{key}_hops', None)
            delay = getattr(calc, f'{key}_delay', None)
            throughput = getattr(calc, f'{key}_throughput', None)
            exec_time = getattr(calc, f'{key}_exec_time', None)
            energy = getattr(calc, f'{key}_energy', None)
            packet_loss = getattr(calc, f'{key}_packet_loss', None)
            print(f"DEBUG metrics_page DB: {algo}: delay={delay}, energy={energy}, packet_loss={packet_loss}")
            if hops is not None or delay is not None or throughput is not None:
                results.append({
                    'algorithm': algo,
                    'hops': hops or 0,
                    'delay': delay or 0,
                    'throughput': throughput or 0,
                    'energy': energy or 0,
                    'packet_loss': packet_loss or 0,
                    'exec_time': exec_time or 0
                })

        source = calc.source
        destination = calc.destination
    else:
        results = session.get('graph_results')
        source = session.get('graph_source')
        destination = session.get('graph_destination')

        if not results or source is None or destination is None:
            return redirect(url_for('input_page'))
    
    # Find best performers (guard against empty results)
    if not results:
        return render_template('metrics.html', error='No metrics available for this calculation')

    # Prefer LoRaWAN when hop counts tie: compute min hops and apply tie-break
    try:
        min_hops = min(x['hops'] for x in results)
    except Exception:
        min_hops = None

    if min_hops is None:
        best_hops_algo = results[0]['algorithm']
    else:
        candidates = [x['algorithm'] for x in results if x['hops'] == min_hops]
        if 'LoRaWAN' in candidates:
            best_hops_algo = 'LoRaWAN'
        else:
            best_hops_algo = candidates[0]
    best_delay_algo = min(results, key=lambda x: x['delay'])['algorithm']
    best_throughput_algo = max(results, key=lambda x: x['throughput'])['algorithm']
    # energy might be missing for some entries; use 0 as fallback
    best_energy_algo = min(results, key=lambda x: x.get('energy', 0))['algorithm']
    best_exec_algo = min(results, key=lambda x: x['exec_time'])['algorithm']
    
    return render_template('metrics.html',
        results=results,
        source=source,
        destination=destination,
        best_hops_algo=best_hops_algo,
        best_delay_algo=best_delay_algo,
        best_throughput_algo=best_throughput_algo,
        best_energy_algo=best_energy_algo,
        best_exec_algo=best_exec_algo
    )


@app.route('/network-graphs', methods=['GET', 'POST'])
@login_required
def network_graphs():
    """Network graph visualization for Flooding vs LoRaWAN"""
    try:
        # Check if coming from calculation or form
        if request.method == 'GET':
            num_nodes = session.get('calc_num_nodes', 25)
            source = session.get('calc_source', 0)
            destination = session.get('calc_destination', 24)
        else:
            num_nodes = int(request.form.get('num_nodes', 25))
            source = int(request.form.get('source', 0))
            destination = int(request.form.get('destination', 24))

        # Validate inputs
        if source >= num_nodes or destination >= num_nodes or source < 0 or destination < 0:
            return render_template('network_graphs.html', 
                                 error='Source/Destination must be less than number of nodes',
                                 num_nodes=num_nodes,
                                 source=source,
                                 destination=destination)

        # Generate network topology
        G = nx.random_geometric_graph(num_nodes, 0.3, seed=42)
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            for node in list(G.nodes()):
                if node not in largest_cc:
                    target = random.choice(list(largest_cc))
                    G.add_edge(node, target)

        # Ensure edges have default weights for metric calculations
        for u, v in G.edges():
            if 'weight' not in G[u][v]:
                G[u][v]['weight'] = 1

        
        # Compute flooding and lorawan algorithms
        try:
            flooding_p, flooding_time = flooding_path(G, source, destination)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            flooding_p, flooding_time = None, 0

        lorawan_p, lorawan_time = energy_aware_path(G, source, destination)

        # Build paths dict for drawing (use empty list if None)
        paths = {
            'Flooding': flooding_p or [],
            'LoRaWAN': lorawan_p or []
        }

        # Generate unique image_id for this calculation to prevent overwrites
        image_id = uuid.uuid4().hex[:8]
        
        # Draw and update graph images used by graph_display.html
        try:
            draw_graph(G, paths, source, destination, image_id)
            draw_heatmap(G, paths, image_id)
        except Exception as e:
            logger.error(f"Drawing graphs failed: {e}")

        # Save results to session so graph_view can display them
        session['graph_source'] = source
        session['graph_destination'] = destination
        session['graph_paths'] = paths
        session['graph_image_id'] = image_id

        # Also store metrics summary for convenience
        results = []
        for name, p, t in [('Flooding', flooding_p, flooding_time), ('LoRaWAN', lorawan_p, lorawan_time)]:
            if p:
                try:
                    # pass the correct algorithm identifier (name) to compute_metrics
                    metrics = compute_metrics(G, p, name)
                except Exception as me:
                    logger.warning(f"compute_metrics failed for {name}: {me}")
                    metrics = {'hops': max(0, len(p)-1), 'delay': 0, 'throughput': 0, 'packet_loss': 0, 'energy': 0}

                results.append({
                    'algorithm': name,
                    'hops': metrics.get('hops', 0),
                    'delay': metrics.get('delay', 0),
                    'throughput': metrics.get('throughput', 0),
                    'packet_loss': metrics.get('packet_loss', 0),
                    'energy': metrics.get('energy', 0),
                    'exec_time': round(t, 3)
                })
        session['graph_results'] = results

        # Prepare interactive vis.js data for Flooding and LoRaWAN views
        flooding_network = {
            'nodes': [{'id': n, 'label': str(n)} for n in G.nodes()],
            'edges': [{'from': u, 'to': v} for u, v in G.edges()]
        }

        lorawan_network = {
            'nodes': [{'id': n, 'label': str(n)} for n in G.nodes()],
            'edges': [{'from': u, 'to': v} for u, v in G.edges()]
        }

        # Compute metrics for flooding and lorawan for comparison
        flooding_metrics = compute_metrics(G, flooding_p, 'Flooding') if flooding_p else {}
        lorawan_metrics = compute_metrics(G, lorawan_p, 'LoRaWAN') if lorawan_p else {}

        # Improvement summary (percentage improvement of LoRaWAN over Flooding)
        def pct_improve(base, new):
            try:
                return round(((base - new) / base) * 100, 2) if base and base != 0 else 0
            except Exception:
                return 0

        improvement = {
            'hops': pct_improve(flooding_metrics.get('hops', 0), lorawan_metrics.get('hops', 0)),
            'delay': pct_improve(flooding_metrics.get('delay', 0), lorawan_metrics.get('delay', 0)),
            'throughput': pct_improve(flooding_metrics.get('throughput', 0), lorawan_metrics.get('throughput', 0)),
            'packet_loss': pct_improve(flooding_metrics.get('packet_loss', 0), lorawan_metrics.get('packet_loss', 0)),
            'energy': pct_improve(flooding_metrics.get('energy', 0), lorawan_metrics.get('energy', 0))
        }

        # Render interactive comparison when requested via GET; otherwise redirect to graph view
        if request.method == 'GET':
            return render_template('network_graphs.html',
                                   graphs_generated=True,
                                   flooding_network=flooding_network,
                                   lorawan_network=lorawan_network,
                                   flooding_path=flooding_p,
                                   lorawan_path=lorawan_p,
                                   flooding_metrics=flooding_metrics,
                                   lorawan_metrics=lorawan_metrics,
                                   improvement=improvement,
                                   num_nodes=num_nodes,
                                   source=source,
                                   destination=destination)

        return redirect(url_for('graph_view'))
    except Exception as e:
        logger.error(f"Network graphs error: {e}")
        # Provide defaults to avoid template undefined errors
        return render_template('network_graphs.html',
                               error=str(e),
                               graphs_generated=False,
                               flooding_network={'nodes': [], 'edges': []},
                               lorawan_network={'nodes': [], 'edges': []},
                               flooding_path=None,
                               lorawan_path=None,
                               flooding_metrics={},
                               lorawan_metrics={},
                               improvement={},
                               num_nodes=session.get('calc_num_nodes', 25),
                               source=session.get('calc_source', 0),
                               destination=session.get('calc_destination', 24))


# ============================
# Routes - Admin Dashboard
# ============================
@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    total_users = User.query.count()
    total_calculations = Calculation.query.count()
    total_logs = AuditLog.query.count()
    recent_logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(20).all()
    
    log_audit(current_user.id, 'ADMIN_ACCESS', 'Admin dashboard accessed')
    
    return render_template('admin_dashboard.html',
        total_users=total_users,
        total_calculations=total_calculations,
        total_logs=total_logs,
        recent_logs=recent_logs
    )


@app.route('/admin/users')
@login_required
def admin_users():
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    users = User.query.all()
    return render_template('admin_users.html', users=users)


@app.route('/admin/promote/<int:user_id>', methods=['POST'])
@login_required
def promote_user(user_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    user = db.session.get(User, user_id)
    if user:
        user.is_admin = True
        db.session.commit()
        log_audit(current_user.id, 'USER_PROMOTION', f'Promoted user: {user.username}')
        return jsonify({'success': True})
    
    return jsonify({'error': 'User not found'}), 404


# ============================
# Routes - Topology Upload
# ============================
@app.route('/upload_topology', methods=['GET', 'POST'])
@login_required
def upload_topology():
    error = None
    success = None
    
    if request.method == 'POST':
        try:
            if 'json_data' in request.form:
                json_data = request.form['json_data']
                G = create_network_from_json(json_data)
                
                if not G:
                    error = "Invalid JSON format"
                else:
                    session['custom_graph'] = json_data
                    success = f"Topology loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
                    log_audit(current_user.id, 'TOPOLOGY_UPLOAD', 'Custom topology uploaded')
            
            elif 'file' in request.files:
                file = request.files['file']
                if file and file.filename.endswith('.json'):
                    json_data = file.read().decode('utf-8')
                    G = create_network_from_json(json_data)
                    
                    if not G:
                        error = "Invalid JSON in file"
                    else:
                        session['custom_graph'] = json_data
                        success = f"Topology loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
                        log_audit(current_user.id, 'TOPOLOGY_UPLOAD', f'File uploaded: {file.filename}')
                else:
                    error = "Please upload a .json file"
        except Exception as e:
            error = f"Error: {str(e)}"
    
    return render_template('upload_topology.html', error=error, success=success)


# ============================
# Routes - API Endpoints
# ============================
@app.route('/api/calculations', methods=['GET', 'POST'])
@login_required
def api_calculations():
    """Get or create calculations"""
    if request.method == 'GET':
        calcs = Calculation.query.filter_by(user_id=current_user.id).all()
        return jsonify([{
            'id': c.id,
            'source': c.source,
            'destination': c.destination,
            'num_nodes': c.num_nodes,
            'created_at': c.created_at.isoformat()
        } for c in calcs])
    
    if request.method == 'POST':
        try:
            data = request.get_json()
            src = int(data.get('source'))
            dst = int(data.get('destination'))
            num_nodes = int(data.get('num_nodes', 25))
            
            G = create_network(num_nodes)
            
            algorithms = {
                'Flooding': flooding_path,
                'LoRaWAN': energy_aware_path
            }
            
            calc = Calculation(user_id=current_user.id, source=src, destination=dst, num_nodes=num_nodes)
            
            for algo_name, algo_func in algorithms.items():
                try:
                    path, exec_time = algo_func(G, src, dst)
                    if path:
                        metrics = compute_metrics(G, path, algo_name)
                        algo = algo_name.lower()
                        setattr(calc, f'{algo}_hops', metrics['hops'])
                        setattr(calc, f'{algo}_delay', metrics['delay'])
                        setattr(calc, f'{algo}_throughput', metrics['throughput'])
                        setattr(calc, f'{algo}_exec_time', exec_time)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
            
            db.session.add(calc)
            db.session.commit()
            
            log_audit(current_user.id, 'API_CALCULATION', f'API calc: {src}->{dst}')
            
            return jsonify({'id': calc.id, 'success': True}), 201
        except Exception as e:
            return jsonify({'error': str(e)}), 400


@app.route('/api/calculations/<int:calc_id>')
@login_required
def api_calculation_detail(calc_id):
    """Get specific calculation"""
    calc = Calculation.query.filter_by(id=calc_id, user_id=current_user.id).first()
    
    if not calc:
        return jsonify({'error': 'Not found'}), 404
    
    return jsonify({
        'id': calc.id,
        'source': calc.source,
        'destination': calc.destination,
        'num_nodes': calc.num_nodes,
        'created_at': calc.created_at.isoformat()
    })


@app.route('/api/stats')
@login_required
def api_stats():
    """Get user statistics"""
    calcs = Calculation.query.filter_by(user_id=current_user.id).all()
    
    if not calcs:
        return jsonify({'total_calculations': 0, 'avg_hops': 0})
    
    return jsonify({
        'total_calculations': len(calcs)
    })


@app.route('/api/audit_logs')
@login_required
def api_audit_logs():
    """Get audit logs (admin only)"""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(100).all()
    return jsonify([{
        'id': l.id,
        'user_id': l.user_id,
        'action': l.action,
        'details': l.details,
        'timestamp': l.timestamp.isoformat(),
        'ip_address': l.ip_address
    } for l in logs])


# ============================
# Advanced Features - Routes
# ============================
@app.route('/network-simulation', methods=['GET', 'POST'])
@login_required
def network_simulation():
    """Simulate network with node failures"""
    if request.method == 'POST':
        try:
            source = int(request.form.get('source', 0))
            destination = int(request.form.get('destination', 0))
            num_nodes = int(request.form.get('num_nodes', 25))
            failed_nodes_str = request.form.get('failed_nodes', '')
            
            # Parse failed nodes
            failed_nodes = [int(x) for x in failed_nodes_str.split(',') if x.strip()]
            
            # Create network and simulate
            G = create_network(num_nodes)
            G_failed = simulate_network_failure(G, failed_nodes)
            
            results = []
            success_count = 0
            total_delay = 0
            
            algorithms = {
                'Flooding': lambda G, s, d: flooding_path(G, s, d),
                'LoRaWAN': energy_aware_path
            }
            
            for algo_name, algo_func in algorithms.items():
                try:
                    path, exec_time = algo_func(G_failed, source, destination)
                    if path:
                        success_count += 1
                        metrics = compute_metrics(G_failed, path, algo_name)
                        total_delay += metrics['delay']
                        results.append({
                            'algorithm': algo_name,
                            'hops': metrics['hops'],
                            'delay': metrics['delay'],
                            'throughput': metrics['throughput'],
                            'exec_time': round(exec_time, 3)
                        })
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
            
            # Save simulation
            success_rate = (success_count / len(algorithms)) * 100 if algorithms else 0
            avg_delay = total_delay / success_count if success_count > 0 else 0
            
            sim = NetworkSimulation(
                user_id=current_user.id,
                name=f"Simulation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                num_nodes=num_nodes,
                failed_nodes=json.dumps(failed_nodes),
                simulation_data=json.dumps(results),
                success_rate=success_rate,
                avg_delay=avg_delay
            )
            db.session.add(sim)
            db.session.commit()
            
            log_audit(current_user.id, 'NETWORK_SIMULATION', 
                     f'Network simulation with {len(failed_nodes)} failed nodes')
            
            return render_template('network_simulation.html', 
                                 results=results, 
                                 success_rate=round(success_rate, 1),
                                 failed_nodes=failed_nodes)
        except Exception as e:
            return render_template('network_simulation.html', error=str(e))
    
    return render_template('network_simulation.html')


@app.route('/energy-routing', methods=['GET', 'POST'])
@login_required
def energy_aware_routing():
    """Energy-aware routing optimization"""
    if request.method == 'POST':
        try:
            source = int(request.form.get('source', 0))
            destination = int(request.form.get('destination', 0))
            num_nodes = int(request.form.get('num_nodes', 25))
            
            G = create_network(num_nodes)
            
            algorithms = {
                'Energy-Aware': energy_aware_path,
                'Multi-Constraint': lambda G, s, d: multi_constrained_path(G, s, d)
            }
            
            results = []
            total_energy = 0
            
            for algo_name, algo_func in algorithms.items():
                path, exec_time = algo_func(G, source, destination)
                if path:
                    metrics = compute_metrics(G, path, algo_name)
                    energy = len(path) * random.uniform(0.8, 1.2)
                    total_energy += energy
                    results.append({
                        'algorithm': algo_name,
                        'hops': metrics['hops'],
                        'delay': metrics['delay'],
                        'energy': round(energy, 2),
                        'throughput': metrics['throughput'],
                        'exec_time': round(exec_time, 3)
                    })
            
            log_audit(current_user.id, 'ENERGY_ROUTING', 
                     f'Energy-aware routing from {source} to {destination}')
            
            return render_template('energy_routing.html',
                                 results=results,
                                 source=source,
                                 destination=destination,
                                 total_energy=round(total_energy, 2))
        except Exception as e:
            return render_template('energy_routing.html', error=str(e))
    
    return render_template('energy_routing.html')


@app.route('/analytics')
@login_required
def analytics():
    """Advanced analytics dashboard"""
    try:
        calcs = Calculation.query.filter_by(user_id=current_user.id).all()
        
        if not calcs:
            return render_template('analytics.html', 
                                 total_calcs=0, 
                                 algorithms_data={})
        
        # Compute statistics for all algorithms
        algorithms = ['flooding', 'lorawan']
        stats = {}
        
        for algo in algorithms:
            hops_list = [getattr(c, f'{algo}_hops') for c in calcs if getattr(c, f'{algo}_hops')]
            delay_list = [getattr(c, f'{algo}_delay') for c in calcs if getattr(c, f'{algo}_delay')]
            throughput_list = [getattr(c, f'{algo}_throughput') for c in calcs if getattr(c, f'{algo}_throughput')]
            
            avg_hops = round(sum(hops_list) / len(hops_list), 2) if hops_list else 0
            avg_delay = round(sum(delay_list) / len(delay_list), 2) if delay_list else 0
            avg_throughput = round(sum(throughput_list) / len(throughput_list), 2) if throughput_list else 0
            
            # Compute efficiency score
            efficiency_score = (1 / (avg_hops + 1)) + (1 / (avg_delay + 1)) + (avg_throughput / 100)
            efficiency_score = round(efficiency_score, 2)
            
            stats[algo] = {
                'avg_hops': avg_hops,
                'avg_delay': avg_delay,
                'avg_throughput': avg_throughput,
                'efficiency_score': efficiency_score,
                'total_runs': len(hops_list)
            }
        
        return render_template('analytics.html',
                             total_calcs=len(calcs),
                             algorithms_data=stats,
                             earliest_calc=calcs[0].created_at if calcs else None,
                             latest_calc=calcs[-1].created_at if calcs else None)
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return render_template('analytics.html', error=str(e))


@app.route('/comparison-graphs')
@login_required
def comparison_graphs():
    """Side-by-side comparison graphs page"""
    return render_template('comparison_graphs.html')


@app.route('/algorithm-comparison')
def algorithm_comparison():
    """Algorithm comparison - Flooding vs LoRaWAN"""
    return render_template('algorithm_comparison.html')


@app.route('/api/comparison-data')
@login_required
def get_comparison_data():
    """Get comparison data for algorithms"""
    try:
        calcs = Calculation.query.filter_by(user_id=current_user.id).all()
        
        if not calcs:
            resp = jsonify({
                'flooding': {'avg_hops': 0, 'avg_delay': 0, 'efficiency_score': 0, 'total_runs': 0},
                'lorawan': {'avg_hops': 0, 'avg_delay': 0, 'efficiency_score': 0, 'total_runs': 0}
            })
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        
        algorithms = ['flooding', 'lorawan']
        comparison_data = {}
        
        for algo in algorithms:
            hops_list = [getattr(c, f'{algo}_hops') for c in calcs if getattr(c, f'{algo}_hops')]
            delay_list = [getattr(c, f'{algo}_delay') for c in calcs if getattr(c, f'{algo}_delay')]
            throughput_list = [getattr(c, f'{algo}_throughput') for c in calcs if getattr(c, f'{algo}_throughput')]
            
            avg_hops = sum(hops_list) / len(hops_list) if hops_list else 0
            avg_delay = sum(delay_list) / len(delay_list) if delay_list else 0
            avg_throughput = sum(throughput_list) / len(throughput_list) if throughput_list else 0
            
            # Efficiency score: lower hops + lower delay + higher throughput
            efficiency_score = (1 / (avg_hops + 1)) + (1 / (avg_delay + 1)) + (avg_throughput / 100)
            
            comparison_data[algo] = {
                'avg_hops': round(avg_hops, 2),
                'avg_delay': round(avg_delay, 2),
                'efficiency_score': round(efficiency_score, 2),
                'total_runs': len(hops_list)
            }
        
        resp = jsonify(comparison_data)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e:
        logger.error(f"Comparison data error: {e}")
        resp = jsonify({'error': str(e)})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 400


@app.route('/debug/comparison-data')
def debug_comparison_data():
    """Debug endpoint: return comparison data for all users (no auth).
    Useful for debugging frontend fetch issues without login/session.
    """
    try:
        calcs = Calculation.query.all()

        if not calcs:
            resp = jsonify({
                'flooding': {'avg_hops': 0, 'avg_delay': 0, 'efficiency_score': 0, 'total_runs': 0},
                'lorawan': {'avg_hops': 0, 'avg_delay': 0, 'efficiency_score': 0, 'total_runs': 0}
            })
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp

        algorithms = ['flooding', 'lorawan']
        comparison_data = {}

        for algo in algorithms:
            hops_list = [getattr(c, f'{algo}_hops') for c in calcs if getattr(c, f'{algo}_hops')]
            delay_list = [getattr(c, f'{algo}_delay') for c in calcs if getattr(c, f'{algo}_delay')]
            throughput_list = [getattr(c, f'{algo}_throughput') for c in calcs if getattr(c, f'{algo}_throughput')]

            avg_hops = sum(hops_list) / len(hops_list) if hops_list else 0
            avg_delay = sum(delay_list) / len(delay_list) if delay_list else 0
            avg_throughput = sum(throughput_list) / len(throughput_list) if throughput_list else 0

            efficiency_score = (1 / (avg_hops + 1)) + (1 / (avg_delay + 1)) + (avg_throughput / 100)

            comparison_data[algo] = {
                'avg_hops': round(avg_hops, 2),
                'avg_delay': round(avg_delay, 2),
                'efficiency_score': round(efficiency_score, 2),
                'total_runs': len(hops_list)
            }

        resp = jsonify(comparison_data)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e:
        logger.error(f"Debug comparison data error: {e}")
        resp = jsonify({'error': str(e)})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 400



@app.route('/service-worker.js')
def service_worker():
    """Serve the service worker from the static folder at the root path.
    This allows the service worker to control the entire site scope.
    """
    return send_from_directory('static', 'service-worker.js')


@app.route('/api/export-results', methods=['POST'])
@login_required
def export_results():
    """Export calculation results as JSON"""
    try:
        calc_id = request.form.get('calc_id')
        calc = Calculation.query.filter_by(id=calc_id, user_id=current_user.id).first()
        
        if not calc:
            return jsonify({'error': 'Not found'}), 404
        
        results = {
            'calculation_id': calc.id,
            'source': calc.source,
            'destination': calc.destination,
            'created_at': calc.created_at.isoformat()
        }
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/history')
@login_required
def history():
    """View calculation history"""
    try:
        calculations = Calculation.query.filter_by(user_id=current_user.id).order_by(
            Calculation.created_at.desc()
        ).all()
        
        return render_template('history.html', calculations=calculations)
    except Exception as e:
        logger.error(f"History error: {e}")
        return render_template('history.html', calculations=[], error=str(e))


@app.route('/api/calculation/<int:calc_id>')
@login_required
def get_calculation(calc_id):
    """Get calculation details including paths"""
    try:
        calc = Calculation.query.filter_by(id=calc_id, user_id=current_user.id).first()
        
        if not calc:
            return jsonify({'error': 'Not found'}), 404
        
        paths_data = []
        for path_obj in calc.paths:
            paths_data.append({
                'algorithm': path_obj.algorithm,
                'path': path_obj.get_path(),
                'path_length': path_obj.path_length
            })
        
        result = {
            'id': calc.id,
            'source': calc.source,
            'destination': calc.destination,
            'num_nodes': calc.num_nodes,
            'created_at': calc.created_at.isoformat(),
            'paths': paths_data
        }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Get calculation error: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/export-calculation/<int:calc_id>')
@login_required
def export_calculation(calc_id):
    """Export calculation as JSON"""
    try:
        calc = Calculation.query.filter_by(id=calc_id, user_id=current_user.id).first()
        
        if not calc:
            return jsonify({'error': 'Not found'}), 404
        
        export_data = {
            'calculation_id': calc.id,
            'source': calc.source,
            'destination': calc.destination,
            'num_nodes': calc.num_nodes,
            'created_at': calc.created_at.isoformat(),
            'paths': []
        }
        
        for path_obj in calc.paths:
            export_data['paths'].append({
                'algorithm': path_obj.algorithm,
                'path': path_obj.get_path(),
                'hops': path_obj.path_length - 1
            })
        
        response = {
            'data': export_data,
            'filename': f"routing_calc_{calc.id}_{calc.created_at.strftime('%Y%m%d_%H%M%S')}.json"
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Export calculation error: {e}")
        return jsonify({'error': str(e)}), 400


# ============================
# Error Handlers
# ============================



# ============================
# Dynamic Routing Video Generation
# ============================
@app.route('/dynamic_routing', methods=['GET'])
@login_required
def dynamic_routing_page():
    """Serve the dynamic routing video generator page"""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dynamic_routing.html')
    with open(html_path, 'r') as f:
        return f.read()


@app.route('/generate_routing_video', methods=['POST'])
@login_required
def generate_routing_video():
    """Generate dynamic routing video based on user parameters"""
    if not HAS_MPL:
        return jsonify({'error': 'Plotting not available. Please install matplotlib/numpy.'}), 400
    
    try:
        data = request.get_json()
        num_nodes = int(data.get('num_nodes', 15))
        source = int(data.get('source', 0))
        destination = int(data.get('destination', 5))
        algorithm_name = data.get('algorithm', 'flooding').lower()
        
        # Validate inputs
        if source >= num_nodes or destination >= num_nodes or source < 0 or destination < 0:
            return jsonify({'error': 'Invalid node numbers'}), 400
        if source == destination:
            return jsonify({'error': 'Source and destination must differ'}), 400
        if num_nodes < 5 or num_nodes > 50:
            return jsonify({'error': 'Number of nodes must be between 5 and 50'}), 400
        
        # Create network
        G = create_network(num_nodes)
        
        # Select algorithm
        algo_map = {
            'energy_aware': energy_aware_path,
        }
        
        # Handle flooding and lorawan separately
        if algorithm_name == 'flooding':
            try:
                if nx.has_path(G, source, destination):
                    path = nx.shortest_path(G, source, destination)
                    exec_time = 0
                else:
                    path = None
                    exec_time = 0
            except Exception as e:
                logger.error(f"Flooding algorithm failed: {e}")
                path = None
                exec_time = 0
        elif algorithm_name == 'lorawan':
            path, exec_time = energy_aware_path(G, source, destination)
        elif algorithm_name in algo_map:
            algo_func = algo_map[algorithm_name]
            path, exec_time = algo_func(G, source, destination)
        else:
            return jsonify({'error': f'Unknown algorithm: {algorithm_name}'}), 400
        
        if not path:
            return jsonify({'error': 'No path found between nodes'}), 400
        
        # Convert path to list of integers for JSON serialization
        path = list(path) if path else []
        path = [int(node) for node in path]
        
        # Compute metrics
        metrics = compute_metrics(G, path, algorithm_name)
        
        # Generate animation frames
        frames = []
        num_frames = 150  # 5s at 30fps
        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            pos = nx.spring_layout(G, seed=42)
        
        for frame_num in range(num_frames):
            fig, ax = plt.subplots(figsize=(12, 9), dpi=100, facecolor='#0a0e27')
            ax.set_facecolor('#0a0e27')
            ax.set_xlim(-0.2, 1.2)
            ax.set_ylim(-0.2, 1.2)
            ax.axis('off')
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, node_color='#00ffff', node_size=400,
                                  edgecolors='#ffffff', linewidths=1.5, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
            nx.draw_networkx_edges(G, pos, width=1, alpha=0.2, edge_color='#e5e7eb', ax=ax)
            
            # Highlight path
            if len(path) > 1:
                edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='#00ff88',
                                      width=3, alpha=0.8, ax=ax)
            
            # Highlight source and destination
            nx.draw_networkx_nodes(G, pos, nodelist=[source], node_color='#10b981',
                                  node_size=600, edgecolors='#047857', linewidths=2, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=[destination], node_color='#ef4444',
                                  node_size=600, edgecolors='#991b1b', linewidths=2, ax=ax)
            
            # Animate packet along path
            progress = frame_num / num_frames
            if len(path) > 1:
                seg_idx = int(progress * (len(path) - 1))
                seg_prog = (progress * (len(path) - 1)) % 1.0
                
                if seg_idx < len(path) - 1:
                    start_node = path[seg_idx]
                    end_node = path[seg_idx + 1]
                    start_pos = pos[start_node]
                    end_pos = pos[end_node]
                    
                    packet_x = start_pos[0] + (end_pos[0] - start_pos[0]) * seg_prog
                    packet_y = start_pos[1] + (end_pos[1] - start_pos[1]) * seg_prog
                    
                    ax.plot(packet_x, packet_y, 'o', color='#00ff88', markersize=12, zorder=100)
            
            # Add title and metrics
            title = f"{algorithm_name.upper()} Routing: {source} → {destination}"
            ax.text(0.5, 1.15, title, transform=ax.transAxes, ha='center', va='top',
                   fontsize=14, color='#ffffff', weight='bold')
            
            metrics_text = f"Hops: {metrics['hops']} | Delay: {metrics['delay']:.1f}ms | Throughput: {metrics['throughput']:.1f} Kbps"
            ax.text(0.5, -0.05, metrics_text, transform=ax.transAxes, ha='center', va='top',
                   fontsize=10, color='#00ffff', family='monospace')
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', facecolor='#0a0e27', bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            img = img.resize((1280, 960), Image.Resampling.LANCZOS)
            frames.append(img)
            plt.close(fig)
        
        # Save as MP4 video
        os.makedirs('static/routing_videos', exist_ok=True)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        mp4_path = f'static/routing_videos/routing_{timestamp}.mp4'
        
        # Use imageio to create MP4 directly (30 FPS)
        import subprocess
        try:
            import imageio
            writer = imageio.get_writer(mp4_path, fps=30, codec='libx264', pixelformat='yuv420p')
            for frame in frames:
                writer.append_data(np.array(frame))
            writer.close()
        except Exception as e:
            # Fallback: save as GIF first, then try ffmpeg conversion
            logger.warning(f"imageio MP4 creation failed: {e}. Trying GIF fallback...")
            gif_path = f'static/routing_videos/routing_{timestamp}.gif'
            frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                          duration=33, loop=0, optimize=False)
            
            # Try to convert with ffmpeg if available
            try:
                subprocess.run(['ffmpeg', '-i', gif_path, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                               '-preset', 'medium', '-b:v', '2000k', '-vf', 'fps=30', 
                               '-movflags', '+faststart', mp4_path],
                              check=True, capture_output=True, timeout=60)
            except Exception as ffmpeg_err:
                logger.error(f"ffmpeg conversion failed: {ffmpeg_err}")
                # Return GIF as fallback
                return jsonify({
                    'video_url': f'/{gif_path}',
                    'preview_url': f'/{gif_path}',
                    'download_url': f'/{gif_path}',
                    'algorithm': algorithm_name,
                    'hops': metrics['hops'],
                    'delay': round(metrics['delay'], 1),
                    'throughput': round(metrics['throughput'], 1),
                    'path': path,
                    'format': 'gif'
                })
        
        log_audit(current_user.id, 'GENERATE_ROUTING_VIDEO',
                 f'Generated video: {source}->{destination} via {algorithm_name}')
        
        return jsonify({
            'video_url': f'/{mp4_path}',
            'download_url': f'/{mp4_path}',
            'preview_url': f'/{mp4_path}',
            'algorithm': algorithm_name,
            'hops': metrics['hops'],
            'delay': round(metrics['delay'], 1),
            'throughput': round(metrics['throughput'], 1),
            'path': path,
            'format': 'mp4'
        })
    
    except Exception as e:
        logger.error(f"Routing video generation error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================
# System Control Routes
# ============================
@app.route('/api/shutdown', methods=['POST'])
@login_required
def shutdown_server():
    """Shutdown the server (admin only)"""
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    log_audit(current_user.id, 'SERVER_SHUTDOWN', 'Server shutdown initiated from mobile')
    
    # Shutdown gracefully
    response = jsonify({'message': 'Server shutting down...', 'status': 'success'})
    
    def shutdown_after_response():
        import sys
        print("\n⛔ Server shutdown initiated from admin panel")
        sys.exit(0)
    
    # Execute shutdown after response is sent
    from threading import Thread
    Thread(target=shutdown_after_response, daemon=False).start()
    
    return response, 200


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        admin_password = os.environ.get('ADMIN_PASSWORD')
        admin_username = os.environ.get('ADMIN_USERNAME', 'admin').strip()
        admin_email = os.environ.get('ADMIN_EMAIL', 'admin@wsn.com').strip()
        if admin_password and not User.query.filter_by(username=admin_username).first():
            admin = User(username=admin_username, email=admin_email, is_admin=True)
            admin.set_password(admin_password)
            db.session.add(admin)
            db.session.commit()
            print(f"Admin user created: {admin_username}")
    
    # Find available port (5000, 5001, 5002, etc.)
    import socket
    port = 5000
    max_attempts = 10
    
    for attempt in range(max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('127.0.0.1', port))
            sock.close()
            break
        except OSError:
            port += 1
    
    print("🚀 Flask Server Starting...")
    print(f"📍 Open your browser and go to: http://127.0.0.1:{port}")
    print(f"📱 WiFi Access: http://192.168.0.107:{port}")
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
