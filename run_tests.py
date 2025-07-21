#!/usr/bin/env python3
"""
Run all tests for the license plate recognition project
"""

import unittest
import sys


def run_all_tests():
    """Run all test suites"""
    print("Running License Plate Recognition Tests")
    print("=" * 50)

    # Add src to path
    sys.path.insert(0, "src")

    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = "tests"
    suite = loader.discover(start_dir, pattern="test_*.py")

    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2, stream=sys.stdout, descriptions=True, failfast=False
    )

    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split()[-1] if traceback else 'Unknown'}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split()[-1] if traceback else 'Unknown'}")

    # Return success status
    success = len(result.failures) == 0 and len(result.errors) == 0

    if success:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


def run_specific_test(test_module):
    """Run a specific test module"""
    print(f"Running {test_module} tests...")

    sys.path.insert(0, "src")

    # Import and run specific test
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f"tests.{test_module}")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return len(result.failures) == 0 and len(result.errors) == 0


def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        # Run specific test
        test_module = sys.argv[1]
        success = run_specific_test(test_module)
    else:
        # Run all tests
        success = run_all_tests() == 0

    # Provide usage information
    if not success or len(sys.argv) == 1:
        print("\nUsage:")
        print("  python run_tests.py                 # Run all tests")
        print("  python run_tests.py test_models     # Run specific test module")
        print("  python run_tests.py test_pipeline   # Run pipeline tests")
        print("  python run_tests.py test_data_loader # Run data loader tests")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
