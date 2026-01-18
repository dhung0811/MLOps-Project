"""
Train CodeBuggy RGCN model và log vào MLflow
"""
import os
import sys
import torch
import joblib
import mlflow
import mlflow.pytorch
from datetime import datetime
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import RGCNConv

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "codebuggy-rgcn"
MODEL_NAME = "codebuggy-detector"


class RGCNDetector(torch.nn.Module):
    """RGCN model for bug detection"""
    def __init__(
        self,
        base_in_dim: int,
        hidden_dim: int,
        num_relations: int,
        num_node_types: int,
        node_type_emb_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.base_in_dim = base_in_dim
        self.node_type_emb_dim = node_type_emb_dim
        self.node_type_emb = torch.nn.Embedding(num_node_types, node_type_emb_dim)
        conv_in_dim = base_in_dim + node_type_emb_dim

        self.convs = torch.nn.ModuleList()
        self.convs.append(RGCNConv(conv_in_dim, hidden_dim, num_relations=num_relations))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations))
        self.dropout = torch.nn.Dropout(dropout)
        self.node_head = torch.nn.Linear(hidden_dim, 1)
        self.graph_head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch

        if x.shape[1] == self.base_in_dim:
            node_type_feats = self.node_type_emb(data.node_type_ids)
            x = torch.cat([x, node_type_feats], dim=1)
        elif x.shape[1] != self.base_in_dim + self.node_type_emb_dim:
            raise ValueError(
                f"Unexpected x dim {x.shape[1]} (expected {self.base_in_dim} or {self.base_in_dim + self.node_type_emb_dim})"
            )

        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = torch.relu(x)
            x = self.dropout(x)
        node_logits = self.node_head(x).squeeze(-1)
        graph_emb = global_mean_pool(x, batch)
        graph_logits = self.graph_head(graph_emb).squeeze(-1)
        return node_logits, graph_logits


def log_model_to_mlflow(
    checkpoint_path: str,
    node_type_path: str = "output/node_type_to_id.joblib",
    mlflow_uri: str = MLFLOW_TRACKING_URI,
):
    """
    Log trained RGCN model vào MLflow
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        node_type_path: Path to node type mapping
        mlflow_uri: MLflow tracking URI
    """
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    
    # Load node type mapping
    if os.path.exists(node_type_path):
        node_type_to_id = joblib.load(node_type_path)
        num_node_types = len(node_type_to_id)
        print(f"Loaded node type mapping: {num_node_types} types")
    else:
        raise FileNotFoundError(f"Node type mapping not found: {node_type_path}")
    
    # Extract model parameters from checkpoint
    base_in_dim = checkpoint.get("base_in_dim")
    hidden_dim = checkpoint.get("hidden_dim")
    relations = checkpoint.get("relations", [])
    num_relations = len(relations)
    
    print(f"\nModel parameters:")
    print(f"  base_in_dim: {base_in_dim}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  num_relations: {num_relations}")
    print(f"  num_node_types: {num_node_types}")
    
    with mlflow.start_run(run_name=f"rgcn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params({
            "base_in_dim": base_in_dim,
            "hidden_dim": hidden_dim,
            "num_relations": num_relations,
            "num_node_types": num_node_types,
            "num_layers": 2,
            "dropout": 0.2,
            "node_type_emb_dim": 64,
        })
        
        # Log final metrics if available
        if "metrics" in checkpoint:
            print(f"\nFinal metrics:")
            for k, v in checkpoint["metrics"].items():
                print(f"  {k}: {v:.4f}")
            mlflow.log_metrics(checkpoint["metrics"])
        
        # Log training history if available
        train_hist_path = "output/train_hist.pt"
        if os.path.exists(train_hist_path):
            print(f"\nLogging training history...")
            training_history = torch.load(train_hist_path, weights_only=False)
            for epoch_data in training_history:
                epoch = epoch_data["epoch"]
                mlflow.log_metrics({
                    "train_loss": epoch_data["train_loss"],
                    "val_loss": epoch_data["val_loss"],
                    "train_node_f1": epoch_data["train_node"][2],
                    "val_node_f1": epoch_data["val_node"][2],
                    "train_graph_f1": epoch_data["train_graph"][2],
                    "val_graph_f1": epoch_data["val_graph"][2],
                }, step=epoch)
            print(f"  Logged {len(training_history)} epochs")
        
        # Recreate model
        print(f"\nRecreating model...")
        model = RGCNDetector(
            base_in_dim=base_in_dim,
            hidden_dim=hidden_dim,
            num_relations=num_relations,
            num_node_types=num_node_types,
            node_type_emb_dim=64,
            num_layers=2,
            dropout=0.2,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        print(f"  Model loaded successfully")
        
        # Log model to MLflow
        print(f"\nLogging model to MLflow...")
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        
        # Log checkpoint file
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
        
        # Log node type mapping
        if os.path.exists(node_type_path):
            mlflow.log_artifact(node_type_path, artifact_path="artifacts")
        
        # Log training history file
        if os.path.exists(train_hist_path):
            mlflow.log_artifact(train_hist_path, artifact_path="artifacts")
        
        run_id = mlflow.active_run().info.run_id
        print(f"\n{'='*80}")
        print(f"✓ Model logged to MLflow!")
        print(f"{'='*80}")
        print(f"Run ID: {run_id}")
        print(f"Model registered as: {MODEL_NAME}")
        print(f"\nNext steps:")
        print(f"1. Promote model to Production:")
        print(f"   python -c \"import mlflow; mlflow.set_tracking_uri('{mlflow_uri}'); ")
        print(f"   client = mlflow.tracking.MlflowClient(); ")
        print(f"   client.transition_model_version_stage(name='{MODEL_NAME}', version='1', stage='Production')\"")
        print(f"\n2. Test inference:")
        print(f"   python model/codebuggy_infer_complete.py")
        
        return run_id


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Log CodeBuggy model to MLflow")
    parser.add_argument("--checkpoint", default="output/rgcn_detector.pt", help="Path to checkpoint")
    parser.add_argument("--node-types", default="output/node_type_to_id.joblib", help="Path to node type mapping")
    parser.add_argument("--mlflow-uri", default=MLFLOW_TRACKING_URI, help="MLflow tracking URI")
    args = parser.parse_args()
    
    log_model_to_mlflow(
        checkpoint_path=args.checkpoint,
        node_type_path=args.node_types,
        mlflow_uri=args.mlflow_uri,
    )
