from dispeecheval import core_loop, add_sqanalyze, io_loop
from argparse import ArgumentDefaultsHelpFormatter
from gooey import Gooey, GooeyParser

def add_filecheck(sparser=None):
    filecheck = sparser.add_parser("check_file", help="Check speech quantity and quality in a file")
    filecheck = sparser.add_parser("check_file", help="Check speech quantity and quality in a file")
    filecheck.add_argument("path", help="Path to audio file", widget="FileChooser")
    filecheck.add_argument("--flag", action="store_true", help="Return audio checker flags")
    filecheck.add_argument("--summarize", action="store_true", help="Return audio checker summaries")
    filecheck.add_argument("--prop_nonspeech_thresh", type=float, default=0.4, help="Flags if the proportion of audio that is non-speech exceeds this threshold")
    filecheck.add_argument("--median_db_diff_thresh", type=float, default=0.065,
                        help="Flags if difference in median speech and non-speech decibel levels as a proportion exceeds this threshold")
    filecheck.add_argument("--print_graph", action="store_true", help="Prints text graph of speech/nonspeech")
    filecheck.add_argument("--print_medians", action="store_true", help="Prints medians")
    return filecheck
    
def add_dircheck(sparser=None):
    dircheck = sparser.add_parser("check_dir", help="Check speech quantity and quality for all audio files in directory")
    dircheck.add_argument("path", help="Path to directory", widget="DirChooser")
    dircheck.add_argument("--flag", action="store_true", help="Return audio checker flags")
    dircheck.add_argument("--summarize", action="store_true", help="Return audio checker summaries")
    dircheck.add_argument("--prop_nonspeech_thresh", type=float, default=0.4, help="Flags if the proportion of audio that is non-speech exceeds this threshold")
    dircheck.add_argument("--median_db_diff_thresh", type=float, default=0.065,
                        help="Flags if difference in median speech and non-speech decibel levels as a proportion exceeds this threshold")
    dircheck.add_argument("--print_graph", action="store_true", help="Prints text graph of speech/nonspeech")
    dircheck.add_argument("--print_medians", action="store_true", help="Prints medians")
    return dircheck

#@Gooey(program_name="DiSpeechEval")
def gui():
    parser = GooeyParser(prog='DiSpeechEval',
                         description='Speech quantity and quality evaluation tool')
    sparser = parser.add_subparsers(help="Function from DiSpeechEval to call")
    filecheck = add_filecheck(sparser)
    dircheck = add_dircheck(sparser)
    sqanalyze = add_sqanalyze(sparser)
    args = parser.parse_args()
    print(args)
    core_loop(args)

@Gooey(program_name="DiSpeechEval")
def gui2():
    parser = GooeyParser(prog='DiSpeechEval',
                         description='Speech quantity and quality evaluation tool')
    io_loop(parser=parser, gui=True)

    

if __name__ == "__main__":
    gui2()