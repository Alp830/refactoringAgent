import sys
import os

print("--- sys.path ---")
for p in sys.path:
    print(p)
print("--- Attempting to import 'code' ---")
try:
    import code
    print("Successfully imported 'code' module.")
    # Try to access root_agent if possible, to further check
    if hasattr(code, 'agent') and hasattr(code.agent, 'root_agent'):
        print("Found code.agent.root_agent.")
    elif hasattr(code, 'root_agent'):
        print("Found code.root_agent.")
    else:
        print("Did not find root_agent directly in 'code' or 'code.agent'.")

except ImportError as e:
    print(f"Failed to import 'code' module: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("--- Script finished ---")
