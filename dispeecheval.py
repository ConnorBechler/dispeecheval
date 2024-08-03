"""
Speech quantity and quality evaluation tool
"""

from pathlib import Path
import librosa
import soundfile
from mosqito import sii_ansi, loudness_zwtv, sharpness_din_tv, roughness_dw
import argparse
from argparse import ArgumentDefaultsHelpFormatter
import rvad_faster
from rVADfast import rVADfast
import time
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from gooey import Gooey

def to_minutes(seconds : int, string=True):
    """Utility for converting seconds to minutes and seconds
    Args:
        seconds (int) : number of seconds
        string (bool) : return output in {minutes}m{seconds}s format (ex. 2m5s)
    Returns:
        EITHER tuple containing minutes in int and seconds in int
        OR string of format {minutes}m{seconds}s
    """
    min = seconds // 60
    sec = seconds % 60
    if string: return(f"{int(min)}m{round(sec)}s")
    else: return(min, sec)

def text_graph(speech_frames : list, length : int, outsize=100):
    """Returns textual representation of presence or absence of phenomenon in an audio file
    Args:
        speech_frames (list) : list of indexes indicating speech labels in an rvad vad_labels output, each index corresponding to 10 ms
        length (int) : length of audio file in seconds
        outsize (int) : number of characters in text graph string output
    Return:
        output string of outsize number of characters
    """
    speech_set = set(speech_frames)
    chunk_len = length*100 / outsize
    output = ""
    for c in range(outsize-1):
        inds = set(range(int(c*chunk_len), int((c+1)*chunk_len)))
        if len(speech_set & inds) > len(inds)/2:
            output += "x"
        else: output+= "-"
    return(output)

def precision(tp, fp): return(tp/(tp+fp))
def recall(tp, fn): return(tp/(tp+fn))
def f1(tp, fp, fn): return((2*tp)/((2*tp)+fp+fn))

def rvadfaster_speech_metrics(lib_aud, sr, rvad_type ="rvad_faster", vad_thres : float =0.4, text_graph_size=100):
    """Uses rVAD to return speech and non-speech sections
    Args:
        lib_aud (np.ndarray) : signal time series
        sr (int) : sampling frequency/sampling rate of signal
        rvad_type (str) : either rvad_faster or rVADfast; rvad_faster is quite a bit faster than rVADfast
        vad_thres (float) : vad threshold used by rvad_faster, included for completeness
        text_graph_size (int) : number of characters in text graph string output
    Returns:
        length_s (int) : total duration in seconds
        speech_s (int) : speech duration in seconds
        nonspeech_s (int) : non-speech duration in seconds
        speech_segments (list) : list of speech segments in [start_time_seconds, end_time_seconds]
        nonspeech_segments (list) : list of non-speech segments in [start_time_seconds, end_time_seconds]
        speech_graph (str) : utility string of xs indicating speech and -s indicating silences, set to 0 to not output
    """
    if rvad_type == "rvad_faster":
        vad_labels = rvad_faster.rVAD_fast(lib_aud, sr, vadThres = vad_thres)
    elif rvad_type == "rVADfast":
        vad=rVADfast(vad_threshold=vad_thres)
        vad_labels = vad(lib_aud, sr)[0]
    win_st, win_end, sil_st, sil_end = None, None, None, None
    speech_segments = []
    nonspeech_segments = []
    speech_frames = []
    nonspeech_frames = []
    for x in range(len(vad_labels)):
        if vad_labels[x] == 0: 
            if win_end != None:
                if win_end != win_st:
                    speech_segments.append([win_st/1000, win_end/1000])
                win_st, win_end = None, None
            nonspeech_frames.append(x)
            if sil_st == None: sil_st = x*10
            sil_end = (x+1)*10
        else : 
            if win_st == None: win_st = x*10
            win_end = (x+1)*10
            speech_frames.append(x)
            if sil_end != None:
                if sil_end != sil_st:
                    nonspeech_segments.append([sil_st/1000, sil_end/1000])
                sil_st, sil_end = None, None
    length_s, speech_s, non_speech_s = len(vad_labels)/100, len(speech_frames)/100, len(nonspeech_frames)/100
    if text_graph_size != 0: speech_graph = text_graph(speech_frames,length_s,outsize=text_graph_size)
    else: speech_graph = ""
    return(length_s, speech_s, non_speech_s, speech_segments, nonspeech_segments, speech_graph)

def return_speech_and_nonspeech_aud(lib_aud : np.ndarray, 
                                    sr : int, 
                                    speech_segments : list, 
                                    nonspeech_segments : list):
    """Splits a time series into two: a speech time-series and a non-speech time series
    Args:
        signal (np.ndarray) : signal time series
        sr (int) : sampling frequency/sampling rate of signal
        speech_segments (list) : list of speech segments in [start_time_seconds, end_time_seconds]
        nonspeech_segments (list) : list of non-speech segments in [start_time_seconds, end_time_seconds]
    Returns:
        speech time series np.ndarray, non-speech time series np.ndarray 
    """
    speech_aud = []
    non_speech_aud = []
    for segment in speech_segments:
        speech_aud += list(lib_aud[librosa.time_to_samples(segment[0], sr=sr):librosa.time_to_samples(segment[1], sr=sr)])
    for segment in nonspeech_segments:
        non_speech_aud += list(lib_aud[librosa.time_to_samples(segment[0], sr=sr):librosa.time_to_samples(segment[1], sr=sr)])
    return(np.array(speech_aud), np.array(non_speech_aud))

def aud_qual_metrics(signal : np.ndarray, 
                     fs : int, 
                     chunk : str, 
                     plot=False,
                     loudness=False,
                     sharpness=False,
                     speechintelindex=False,
                     roughness=False):
    """Applies selected sound quality metrics to a given signal using MOSQITO package
    Args:
        signal (np.ndarray) : signal time series
        fs (int) : sampling frequency/sampling rate of signal
        chunk (str) : string name of sample for display outputs
        plot (bool) : save plots of loudness, sii, and roughness?
        loudness (bool) : calculate Zwicker time varying loudness?
        sharpness (bool) : calculate sharpness of time varying signal?
        speechintelindex (bool) : calculate ANSI speech intelligibility index for signal?
        roughness (bool) : calculate Daniel and Weber method roughness for signal?
    Returns:
        list of tuple sound quality calculation results, in order of calculation arguments
            i.e., if all are calculated, order is loudness results, sharpness results, sii results, roughness results
    """
    results = []
    print(chunk, end=" ")
    if loudness:
        st = time.time()
        #N, N_spec, bark_axis, time_axis = loudness_zwtv(signal, fs)
        loud_met = loudness_zwtv(signal, fs)
        results.append(loud_met)
        print("loudness", loud_met[0], time.time()-st)
        if plot:
            plt.plot(loud_met[3], loud_met[0])
            plt.xlabel("Time [s]")
            plt.ylabel("Loudness [Sone]")
            plt.savefig(f"{chunk}_loudness.png")
            plt.clf()
    if sharpness:
        st = time.time()
        shrp_met = sharpness_din_tv(signal, fs)
        results.append(shrp_met)
        print("sharpness",shrp_met[0], time.time()-st)
    if speechintelindex:
        st = time.time()
        #SII, SII_spec, freq_axis = sii_ansi(signal, fs, method='critical', speech_level='normal')
        sii_met = sii_ansi(signal, fs, method='critical', speech_level='normal')
        results.append(sii_met)
        print("speech_intel",sii_met[0], time.time()-st)
        if plot:
            plt.plot(sii_met[2], sii_met[1])
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Specific value ")
            plt.title(f"Speech Intelligibility Index = " + f"{sii_met[0]:.2f}")
            plt.savefig(f"{chunk}_sii.png")
            plt.clf()
        st = time.time()
    if roughness:
        #R, R_spec, bark, time_ax = roughness_dw(signal, fs)
        roug_met = roughness_dw(signal, fs)
        results.append(roug_met)
        print("Roughness", roug_met[0][0], time.time() - st)
        if plot:
            plt.plot(roug_met[2], roug_met[1])
            plt.xlabel("Bark axis [Bark]")
            plt.ylabel("Specific roughness [Asper/Bark]")
            plt.title("Roughness = " + f"{roug_met[0][0]:.2f}" + " [Asper]")
            plt.savefig(f"{chunk}.png")
    return(results)

def check_audio(path, 
         return_flags = True,
         return_summary = False,
         prop_nonspeech_thresh = .4, 
         median_db_diff_thresh = .065, 
         print_graph=False, 
         print_medians=False):
    """
    Function for flagging audio files whose speech is either too short relative to the length of the recording 
    or too close in volume to non-speech
    Args:
        path (pathlib.Path) : path to an audio file supported by soundfile or a directory with said files
        prop_nonspeech_thresh (float) : the proportion of audio that is non-speech threshold, flags if greater than this argument
        median_db_diff_thresh (float) : difference in median speech and non-speech decibel levels 
            as a proportion of total recording decibel range threshold, flags if less than this argument
        print_graph (bool) : prints text graph of speech to non-speech in the audio file if True
        print_medians (bool) : prints median decibel levels for speech and non-speech in the audio file, as well as
            the difference between the two as a proportion of the audio file's decibel range if True
    Returns:
        results (polars.DataFrame) : table of varying dimensions depending on arguments, starts with file name
        if return_flags is True, silence_flag (bool) and db_diff_flag (bool) are the first two items in the list
        if return_summary is True, length (int), speech (int), nonspeech (int), sp_median_db (float), 
            nsp_median_db (float), and db_range (float) are added to the results
    """
    results = []
    results_colheads = ["File"]
    if return_flags: results_colheads += ["nonspeech_flag", "db_diff_flag"]
    if return_summary: results_colheads += ["total_s","speech_s","nonspeech_s", "speech_median_db","nonspeech_median_db","db_range"]
    soundfile.available_formats
    if path.suffix == "": 
        paths = [path for path in path.iterdir() if path.suffix[1:].upper() in soundfile.available_formats().keys()]
    else: paths = [path]
    for path in paths:
        result = [path.stem]
        sample, sr = librosa.load(path)
        length, speech, nonspeech, segments, nonspeech_segments, graph = rvadfaster_speech_metrics(sample, sr)
        db_aud = librosa.amplitude_to_db(sample)
        db_max, db_min = np.min(db_aud), np.max(db_aud)
        db_range = abs(db_max-db_min)
        speech_aud, nonspeech_aud = return_speech_and_nonspeech_aud(db_aud, sr, segments, nonspeech_segments)
        sp_median_db, nsp_median_db = np.median(speech_aud), np.median(nonspeech_aud)
        #print("Speech", to_minutes(speech), "Non-Speech", to_minutes(nonspeech))
        if print_graph: print(graph)
        if print_medians: 
                print("Median speech db:",sp_median_db,
                    "Median non-speech db:",nsp_median_db,
                    "Diff %", (sp_median_db-nsp_median_db)/db_range)
        if return_flags:
            silence_flag, db_diff_flag = False, False
            if nonspeech/length > prop_nonspeech_thresh: 
                silence_flag = True
            if (sp_median_db - nsp_median_db)/db_range < median_db_diff_thresh :
                db_diff_flag = True
            result += [silence_flag, db_diff_flag]
        if return_summary: 
            result += [length, speech, nonspeech, sp_median_db, nsp_median_db, db_range]
        results.append(result)
    table = pl.DataFrame(results,results_colheads,orient="row")
    return(table) 

def cli():
    parser = argparse.ArgumentParser(prog='DiSpeechEval',
                                     description='Speech quantity and quality evaluation tool',
                                     formatter_class=ArgumentDefaultsHelpFormatter)#,
                                     #epilog='Text at the bottom of help')
    parser.add_argument("path")
    parser.add_argument("--flag", action="store_true", help="Return audio checker flags")
    parser.add_argument("--summarize", action="store_true", help="Return audio checker summaries")
    parser.add_argument("--prop_nonspeech_thresh", type=float, default=0.4, help="Flags if the proportion of audio that is non-speech exceeds this threshold")
    parser.add_argument("--median_db_diff_thresh", type=float, default=0.065,
                        help="Flags if difference in median speech and non-speech decibel levels as a proportion exceeds this threshold")
    parser.add_argument("--print_graph", action="store_true", help="Prints text graph of speech/nonspeech")
    parser.add_argument("--print_medians", action="store_true", help="Prints medians")
    args = vars(parser.parse_args())

    print(check_audio(path=Path(args["path"]), 
                      return_flags=args["flag"],
                      return_summary=args["summarize"],
                      prop_nonspeech_thresh=args["prop_nonspeech_thresh"],
                      median_db_diff_thresh=args["median_db_diff_thresh"],
                      print_graph=args["print_graph"],
                      print_medians=args["print_medians"]))

#First Run: Median Speech DB Diff < .065, Non-speech Prop Thresh > .4
#8 True Positives, 9 False Positives, 0 False Negatives
#print(precision(8, 9), recall(8, 0), f1(8,9,0))

if __name__ == "__main__":        
    cli()
    """
    #dir = Path("C:/Users/bechl/Downloads/low_aud_qual")
    dirs = [Path("C:/Users/bechl/Downloads/kid"),Path("C:/Users/bechl/Downloads/teen"),Path("C:/Users/bechl/Downloads/adult")]
    st = time.time()
    #results = check_audio(dirs[2].joinpath("MCD-00157_2024-02-02_01_clear.wav"),return_summary=True)
    #print(results)
    #print("Took",time.time()-st, "seconds")
    if True:
        total_length = 0
        for dir in dirs:
            print(dir)
            paths = [path for path in dir.iterdir() if path.suffix in [".wav", ".mp3"]]
            for path in paths:
                print(path.stem, end=" ")
                results = check_audio(path, return_summary=True)
                print(path.stem, "Flagged for Non-Speech :",results[0],"Flagged for DB diff:", results[1])
                total_length += results[2]
    print("Took",time.time()-st, "seconds,", to_minutes(time.time()-st, string=True))
    print("Total audio length was", total_length, "seconds", to_minutes(total_length, string=True))
    if False:
        graphs = ""
        col_names = ["file","length(s)","speech(s)","prop_nonspeech","SII", "db_speech","db_nonspeech"]
        col_vals = []
        for path in paths:
            print(path.stem)
            sample, sr = librosa.load(path)
            #length, speech, nonspeech, chunks, graph = rvadfast_speech_metrics(sample, sr, name=path.stem)
            st = time.time()
            length, speech, nonspeech, chunks, ns_chunks, graph = rvadfaster_speech_metrics(sample, sr, name=path.stem)
            print("Getting rvad segments and aud data took", time.time()-st, "seconds")
            st = time.time()
            db_aud = librosa.amplitude_to_db(sample)
            db_max, db_min = np.min(db_aud), np.max(db_aud)
            db_range = abs(db_max-db_min)
            speech_aud, nonspeech_aud = return_speech_and_nonspeech_aud(db_aud, sr, chunks, ns_chunks)
            print("Converting aud to db and returning speech/nonspeech took", time.time()-st, "seconds")
            #speech_aud, nonspeech_aud = librosa.amplitude_to_db(speech_aud), librosa.amplitude_to_db(nonspeech_aud)
            st = time.time()
            sp_median_db, nsp_median_db = np.median(speech_aud), np.median(nonspeech_aud)
            print("Calculating median db took", time.time()-st, "seconds")
            #print("speech", sp_median_db/db_range, "nonspeech", nsp_median_db/db_range, "diff", (sp_median_db/db_range)-(nsp_median_db/db_range))
            graphs += graph + path.stem + "\n"
            if True:
                print("total length", to_minutes(length))
                print("speech", to_minutes(speech), end=" ")
                print("nonspeech", to_minutes(nonspeech))
                print("prop nonspeech", nonspeech/length)
                print("median speech amplitude", sp_median_db)
                print("median nonspeech amplitude", nsp_median_db)
                SII = "NOT RETURNED"#aud_qual_metrics(sample, sr, chunk=path.stem)
                col_vals.append([path.stem, length, speech, nonspeech/length, SII, sp_median_db/db_range, nsp_median_db/db_range])
                #print(chunks)
                if False:
                    for c, chunk in enumerate(chunks[:-1]):
                        data = sample[librosa.time_to_samples(chunk[0], sr=sr):
                                    librosa.time_to_samples(chunk[1], sr=sr)]
                        #print(data)
                        print(f"speech {chunk}", np.average(abs(data)))
                        sil_chunk = [chunk[1], chunks[c+1][0]]
                        sil_data = sample[librosa.time_to_samples(sil_chunk[0], sr=sr)+1:
                                    librosa.time_to_samples(sil_chunk[1], sr=sr)]
                        print(f"nonspeech {sil_chunk}", np.average(abs(sil_data)))
                        plt.plot(data)
                        plt.title(f"abs db ={np.average(abs(data))}")
                        plt.savefig(f"{chunk}speech.png")
                        plt.clf()
                        plt.plot(sil_data)
                        plt.title(f"abs db ={np.average(abs(sil_data))}")
                        plt.savefig(f"{sil_chunk}nonspeech.png")
                        plt.clf()
                        #soundfile.write(f"speech{chunk}.mp3",data,sr)
                        #soundfile.write(f"nspeech{sil_chunk}.mp3",sil_data,sr)
                        #aud_qual_metrics(data, sr, chunk)
        out_table = pl.DataFrame(col_vals, col_names,orient="row")
        out_table.write_csv(dir.joinpath(f"{dir.stem}_summary_table_avs.csv"))
        print(graphs)
    """
    