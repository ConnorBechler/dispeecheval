"""
Diary evaluation tool
"""

from pathlib import Path
import librosa
import soundfile
from mosqito import *
import argparse
import rvad_faster
from math import ceil
import os

def rvad_chunk_faster(lib_aud, min_chunk, max_chunk, sr, stride):
    """Uses rvad_faster to chunk audio, with fall-back stride chunking for segments that are too long"""
    aud_mono = librosa.to_mono(lib_aud)
    soundfile.write("rvad_working_mono.wav", aud_mono, sr)
    segs = rvad_faster.rVAD_fast("rvad_working_mono.wav", ftThres = 0.4)
    os.remove("rvad_working_mono.wav")
    win_st = None
    win_end = None
    nchunks = []
    for x in range(len(segs)):
        if segs[x] == 1:
            if win_st == None: win_st = x*10
            win_end = x*10
        elif segs[x] == 0:
            if win_end != None:
                diff = win_end - win_st
                if diff > max_chunk:
                    num_chunks = ceil(diff/max_chunk)
                    step = round(diff/num_chunks)
                    newest_chunks = [[win_st+(step*x)-stride, win_st+(step*(x+1))+stride, 
                                    (stride, stride)] for x in range(num_chunks)]
                    if stride > 0 :
                        newest_chunks[0] = [win_st, win_st+(step+stride), (0, stride)]
                        newest_chunks[-1] = [win_end-step-stride, win_end, (stride, 0)]
                    nchunks += newest_chunks
                elif diff > min_chunk:
                    nchunks.append([win_st, win_end, (0, 0)])
                win_st, win_end = None, None
    return(nchunks)