import sys
import os

# Add the project root to the python path so imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.reasoning_agent import SecurityAnalyst

def start_chat():
    print("\n" + "="*50)
    print("   CONTEXT-AWARE SURVEILLANCE ANALYST (AI)   ")
    print("="*50)
    print("Initializing System... Please wait.")
    
    try:
        bot = SecurityAnalyst()
        print("\n✅ Analyst Ready. You can ask questions like:")
        print("   - 'Who entered the room?'")
        print("   - 'Was there any theft today?'")
        print("   - 'Did Talha access the zone?'")
        print("Type 'exit' or 'quit' to stop.")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Failed to initialize AI: {e}")
        return

    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting. Stay Safe!")
                break
                
            if not user_input:
                continue
                
            print("Analyst: Thinking...", end="\r")
            
            # Get answer from the agent
            response = bot.consult(user_input)
            
            # Clear "Thinking..." and print response
            print(" " * 20, end="\r") 
            print(f"Analyst: {response}")
            
        except KeyboardInterrupt:
            print("\nExiting. Stay Safe!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    start_chat()