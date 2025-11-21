#!/usr/bin/env python3
"""
Simple test script to verify all imports work correctly
"""

def test_imports():
    """Test all the required imports for Part 2"""
    print("Testing imports...")
    
    try:
        # Test basic imports
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        # Test transformers
        from transformers import T5ForConditionalGeneration, T5TokenizerFast
        print("✅ Transformers T5 classes")
        
        # Test our modules
        from utils import compute_metrics, save_queries_and_records
        print("✅ utils.py imports")
        
        from load_data import T5Dataset, load_t5_data
        print("✅ load_data.py imports")
        
        from t5_utils import initialize_model
        print("✅ t5_utils.py imports")
        
        # Test tokenizer
        tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        print("✅ T5 tokenizer loads successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_files():
    """Test if required data files exist"""
    import os
    
    required_files = [
        'data/train.nl',
        'data/train.sql', 
        'data/dev.nl',
        'data/dev.sql',
        'data/test.nl',
        'data/flight_database.db'
    ]
    
    print("\nChecking data files...")
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} required data files")
        return False
    else:
        print("\n✅ All required data files found")
        return True

if __name__ == "__main__":
    print("🧪 Part 2 Environment Test")
    print("=" * 40)
    
    imports_ok = test_imports()
    data_ok = test_data_files()
    
    print("\n" + "=" * 40)
    if imports_ok and data_ok:
        print("✅ Environment test PASSED - Ready for Part 2!")
    else:
        print("❌ Environment test FAILED - Please fix issues above")
        if not imports_ok:
            print("  - Fix import issues")
        if not data_ok:
            print("  - Ensure data directory exists with required files")