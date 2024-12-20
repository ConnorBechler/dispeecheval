from dispeecheval import add_sqanalyze, io_loop
from argparse import ArgumentDefaultsHelpFormatter
from gooey import Gooey, GooeyParser

@Gooey(program_name="DiSpeechEval")
def gui():
    parser = GooeyParser(prog='DiSpeechEval',
                         description='Speech quantity and quality evaluation tool')
    io_loop(parser=parser, gui=True)

    
if __name__ == "__main__":
    gui()