from dispeecheval import cli
from gooey import Gooey

@Gooey
def gui():
    cli()

if __name__ == "__main__":        
    gui()