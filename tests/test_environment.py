"""
Environment Test Script
Verifies all core packages work correctly
"""


def test_imports():
    """Test all critical imports"""
    print("🔍 Testing core package imports...")

    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")

        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")

        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")

        import requests
        print(f"✅ Requests {requests.__version__}")

        from dotenv import load_dotenv
        print("✅ Python-dotenv")

        import joblib
        print(f"✅ Joblib {joblib.__version__}")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_numpy_operations():
    """Test numpy operations work correctly"""
    print("\n🔬 Testing NumPy operations...")

    try:
        import numpy as np

        # Basic operations
        arr = np.array([1, 2, 3, 4, 5])
        result = np.mean(arr)
        print(f"✅ Array mean: {result}")

        # Random operations
        random_data = np.random.normal(0, 1, 100)
        print(f"✅ Random data shape: {random_data.shape}")

        # Linear algebra
        matrix = np.array([[1, 2], [3, 4]])
        inv_matrix = np.linalg.inv(matrix)
        print(f"✅ Matrix inversion works")

        return True

    except Exception as e:
        print(f"❌ NumPy operation failed: {e}")
        return False


def test_pandas_operations():
    """Test pandas operations work correctly"""
    print("\n📊 Testing Pandas operations...")

    try:
        import pandas as pd
        import numpy as np

        # Create sample DataFrame
        df = pd.DataFrame({
            'city': ['London', 'Tokyo', 'Delhi'],
            'pm25': [15.2, 25.8, 45.1],
            'temperature': [22.0, 18.5, 35.2]
        })

        print(f"✅ DataFrame created: {df.shape}")
        print(f"✅ DataFrame operations work")

        # Test CSV operations
        import os
        os.makedirs('tests/temp', exist_ok=True)
        df.to_csv('tests/temp/test.csv', index=False)
        df_loaded = pd.read_csv('tests/temp/test.csv')
        print(f"✅ CSV read/write works")

        # Cleanup
        os.remove('tests/temp/test.csv')
        os.rmdir('tests/temp')

        return True

    except Exception as e:
        print(f"❌ Pandas operation failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 TESTING ENVIRONMENT SETUP")
    print("=" * 40)

    tests = [
        test_imports,
        test_numpy_operations,
        test_pandas_operations
    ]

    results = []
    for test in tests:
        results.append(test())
        print()

    if all(results):
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your environment is ready for MLOps development")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Please fix the failing components before proceeding")
        return False


if __name__ == "__main__":
    success = main()
