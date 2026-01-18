"""
CodeBuggy Example Usage
Demonstrates 3 types of bugs: Array Index, Null Pointer, Resource Leak
"""

# Example 1: Array Index Out of Bounds
BUGGY_ARRAY_INDEX = """
public int sum(int[] arr) {
    int s = 0;
    for (int i = 0; i <= arr.length; i++) {
        s += arr[i];
    }
    return s;
}
""".strip()

FIXED_ARRAY_INDEX = """
public int sum(int[] arr) {
    int s = 0;
    for (int i = 0; i < arr.length; i++) {
        s += arr[i];
    }
    return s;
}
""".strip()

# Example 2: Null Pointer Dereference
BUGGY_NULL_POINTER = """
public String getName(User user) {
    return user.getName().toUpperCase();
}
""".strip()

FIXED_NULL_POINTER = """
public String getName(User user) {
    if (user == null || user.getName() == null) {
        return null;
    }
    return user.getName().toUpperCase();
}
""".strip()

# Example 3: Resource Leak
BUGGY_RESOURCE_LEAK = """
public String readFile(String path) throws IOException {
    FileReader fr = new FileReader(path);
    BufferedReader br = new BufferedReader(fr);
    return br.readLine();
}
""".strip()

FIXED_RESOURCE_LEAK = """
public String readFile(String path) throws IOException {
    try (FileReader fr = new FileReader(path);
         BufferedReader br = new BufferedReader(fr)) {
        return br.readLine();
    }
}
""".strip()


def main():
    from codebuggy_infer_complete import CodeBuggyPredictor
    
    print("="*80)
    print("CodeBuggy Example - 3 Bug Types")
    print("="*80)
    
    # Initialize predictor
    predictor = CodeBuggyPredictor(
        mlflow_uri="http://localhost:5000",
        model_name="codebuggy-detector",
        model_stage="Production",
    )
    
    examples = [
        ("Array Index Out of Bounds", BUGGY_ARRAY_INDEX, FIXED_ARRAY_INDEX),
        ("Null Pointer Dereference", BUGGY_NULL_POINTER, FIXED_NULL_POINTER),
        ("Resource Leak", BUGGY_RESOURCE_LEAK, FIXED_RESOURCE_LEAK),
    ]
    
    for bug_type, buggy, fixed in examples:
        print(f"\n{'='*80}")
        print(f"Example: {bug_type}")
        print(f"{'='*80}")
        print(f"\nBuggy Code:")
        print(buggy)
        print(f"\nFixed Code:")
        print(fixed)
        
        # Predict
        results = predictor.predict(buggy, fixed, log_to_mlflow=False)
        
        input("\nPress Enter to continue to next example...")


if __name__ == "__main__":
    main()
