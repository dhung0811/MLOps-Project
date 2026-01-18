"""
Complete CodeBuggy Inference Script
Input: 2 strings (buggy_function, fixed_function)
Output: Bug predictions at node and graph level
"""
import os
import sys
import torch
import numpy as np
import joblib
import mlflow
import mlflow.pytorch
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data
from datetime import datetime

# Add utils to path
sys.path.append(os.path.dirname(__file__))
from utils.build_graph import (
    build_ast_graph,
    build_graph_parts,
    compute_code_embeddings,
    format_top_predictions,
    get_node_type_id
)

class CodeBuggyPredictor:
    def __init__(
        self,
        mlflow_uri: str = "http://localhost:5000",
        model_name: str = "codebuggy-detector",
        model_stage: str = "Version 3",
        graphcodebert_model: str = "microsoft/graphcodebert-base",
    ):
        self.mlflow_uri = mlflow_uri
        self.model_name = model_name
        self.model_stage = model_stage
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Device: {self.device}")
        
        # Load GraphCodeBERT
        print(f"Loading {graphcodebert_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(graphcodebert_model)
        
        # Load model with proper device handling to avoid meta tensor issues
        self.encoder = AutoModel.from_pretrained(
            graphcodebert_model,
            torch_dtype=torch.float32,  # Explicit dtype
            low_cpu_mem_usage=False,     # Disable meta device
        )
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        print(f"✓ GraphCodeBERT loaded")
        
        # Load model from MLflow
        self.model = None
        self.load_model_from_mlflow()
    
    def load_model_from_mlflow(self):
        """Load model from MLflow Registry"""
        mlflow.set_tracking_uri(self.mlflow_uri)
        
        print(f"\nLoading model from MLflow...")
        print(f"  Model: {self.model_name}")
        print(f"  Stage: {self.model_stage}")
        
        model_uri = f"models:/{self.model_name}/{self.model_stage}"
        
        try:
            self.model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
            self.model.eval()
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print(f"  Trying latest version...")
            try:
                model_uri = f"models:/{self.model_name}/latest"
                self.model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
                self.model.eval()
                print(f"✓ Loaded latest version")
            except Exception as e2:
                raise RuntimeError(f"Failed to load model: {e2}")

    def build_graph(self, buggy_code: str, fixed_code: str):
        """Build graph from buggy and fixed code"""
        # Build AST
        # nodes, parents, children, edges, edge_types = build_ast_graph(buggy_code)
        parts = build_graph_parts(buggy_code, fixed_code, method_id=None)
        
        # Compute code embeddings
        code_embs = compute_code_embeddings(
            parts["buggy_code"], parts["nodes"], parts["children"],
            self.tokenizer, self.encoder, self.device
        )
        
        x = torch.cat([code_embs, parts["diff_feats"]], dim=1)
        
        # Node type IDs
        node_type_ids = torch.tensor(
            [get_node_type_id(n["node_type"]) for n in parts["nodes"]],
            dtype=torch.long
        )
        
        # Edge index and types
        if parts["edges"]:
            edge_index = torch.tensor(parts["edges"], dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(parts["edge_types"], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)
        
        # Create PyG Data
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            node_type_ids=node_type_ids,
        )
        
        return data, parts
    
    def predict(self, buggy_code: str, fixed_code: str, log_to_mlflow: bool = False):
        """
        Predict bugs in code
        
        Args:
            buggy_code: Source code with bugs
            fixed_code: Fixed version of code
            log_to_mlflow: Whether to log results to MLflow
            
        Returns:
            dict with predictions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        print(f"\n{'='*80}")
        print(f"CodeBuggy Inference")
        print(f"{'='*80}")
        
        # Build graph
        print("Building graph...")
        try:
            data, parts = self.build_graph(buggy_code, fixed_code)
            data = data.to(self.device)
            
            print(f"  Nodes: {len(parts["nodes"])}")
            print(f"  Edges: {data.edge_index.shape[1]}")
            print(f"  Feature shape: {data.x.shape}")
        except Exception as e:
            print(f"Error building graph: {e}")
            raise
        
        # Predict
        print("Running inference...")
        try:
            with torch.inference_mode():
                # Add batch dimension if needed
                if not hasattr(data, 'batch'):
                    import torch_geometric
                    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)
                
                node_logits, graph_logits = self.model(data)
            
            # Handle output shapes
            node_probs = torch.sigmoid(node_logits).squeeze().detach().cpu().numpy()
            graph_prob = float(torch.sigmoid(graph_logits).squeeze().item())
            
            print(f"  Node logits shape: {node_logits.shape}")
            print(f"  Graph logits shape: {graph_logits.shape}")
        except Exception as e:
            print(f"Error during model inference: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Format results
        print(f"\n{'='*80}")
        print(f"Results")
        print(f"{'='*80}")
        print(f"Graph Bug Probability: {graph_prob:.4f}")
        
        try:
            print(format_top_predictions(node_probs, parts["nodes"], buggy_code, top_k=10))
        except Exception as e:
            print(f"Error formatting predictions: {e}")
            # Continue anyway, we still have the results
        
        results = {
            "graph_probability": graph_prob,
            "node_probabilities": node_probs,
            "nodes": parts["nodes"],
            "num_nodes": len(parts["nodes"]),
            "num_edges": data.edge_index.shape[1],
        }
        
        # Log to MLflow
        if log_to_mlflow:
            self._log_to_mlflow(buggy_code, fixed_code, results)
        
        return results
    
    def _log_to_mlflow(self, buggy_code: str, fixed_code: str, results: dict):
        """Log inference results to MLflow"""
        import tempfile
        
        mlflow.set_tracking_uri(self.mlflow_uri)
        
        with mlflow.start_run(run_name=f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params({
                "model_name": self.model_name,
                "model_stage": self.model_stage,
                "num_nodes": results["num_nodes"],
                "num_edges": results["num_edges"],
            })
            
            # Log metrics
            mlflow.log_metrics({
                "graph_probability": results["graph_probability"],
                "max_node_probability": float(results["node_probabilities"].max()),
                "mean_node_probability": float(results["node_probabilities"].mean()),
            })
            
            # Log code using temporary files
            with tempfile.NamedTemporaryFile(mode="w", suffix=".java", delete=False) as f:
                f.write(buggy_code)
                buggy_path = f.name
            
            with tempfile.NamedTemporaryFile(mode="w", suffix=".java", delete=False) as f:
                f.write(fixed_code)
                fixed_path = f.name
            
            try:
                mlflow.log_artifact(buggy_path, artifact_path="code/buggy_code.java")
                mlflow.log_artifact(fixed_path, artifact_path="code/fixed_code.java")
                print(f"\n✓ Results logged to MLflow (Run ID: {mlflow.active_run().info.run_id})")
            finally:
                # Clean up temporary files
                import os
                try:
                    os.unlink(buggy_path)
                    os.unlink(fixed_path)
                except:
                    pass


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CodeBuggy Inference")
    parser.add_argument("--buggy", type=str, help="Buggy function code")
    parser.add_argument("--fixed", type=str, help="Fixed function code")
    parser.add_argument("--buggy-file", type=str, help="File containing buggy code")
    parser.add_argument("--fixed-file", type=str, help="File containing fixed code")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    parser.add_argument("--model-name", default="codebuggy-detector")
    parser.add_argument("--model-stage", default="Production")
    parser.add_argument("--log-mlflow", action="store_true", help="Log results to MLflow")
    args = parser.parse_args()
    
    # Get code from args or files
    if args.buggy_file and args.fixed_file:
        with open(args.buggy_file) as f:
            buggy_code = f.read()
        with open(args.fixed_file) as f:
            fixed_code = f.read()
    elif args.buggy and args.fixed:
        buggy_code = args.buggy
        fixed_code = args.fixed
    else:
        # Example code
        buggy_code = """
public int sum(int[] arr) {
    int s = 0;
    for (int i = 0; i <= arr.length; i++) {
        s += arr[i];
    }
    return s;
}
""".strip()
        
        fixed_code = """
public int sum(int[] arr) {
    int s = 0;
    for (int i = 0; i < arr.length; i++) {
        s += arr[i];
    }
    return s;
}
""".strip()
        
        print("Using example code (provide --buggy and --fixed for custom input)")
    
    # Initialize predictor
    try:
        predictor = CodeBuggyPredictor(
            mlflow_uri=args.mlflow_uri,
            model_name=args.model_name,
            model_stage=args.model_stage,
        )
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Predict
    try:
        results = predictor.predict(buggy_code, fixed_code, log_to_mlflow=args.log_mlflow)
        print(f"\n{'='*80}")
        print("Inference completed successfully!")
        print(f"{'='*80}")
        return 0
    except Exception as e:
        print(f"\nError during prediction: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
