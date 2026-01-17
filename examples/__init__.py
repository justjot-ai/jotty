"""Example integrations and wrappers"""


class HelloWorld:
    """A reusable Hello World class demonstrating basic Python structure."""

    def __init__(self, greeting: str = "Hello, World!"):
        """
        Initialize HelloWorld with a custom greeting.

        Args:
            greeting: The greeting message to display (default: "Hello, World!")
        """
        self.greeting = greeting

    def greet(self) -> str:
        """
        Return the greeting message.

        Returns:
            The greeting string
        """
        return self.greeting

    def print_greeting(self) -> None:
        """Print the greeting message to stdout."""
        print(self.greeting)


def hello_world(message: str = "Hello, World!") -> None:
    """
    Simple function to print a hello world message.

    Args:
        message: The message to print (default: "Hello, World!")
    """
    print(message)


if __name__ == "__main__":
    # Demonstrate class-based approach
    greeter = HelloWorld()
    greeter.print_greeting()

    # Demonstrate function-based approach
    hello_world()

    # Demonstrate customization
    custom_greeter = HelloWorld("Hello from Jotty!")
    custom_greeter.print_greeting()
