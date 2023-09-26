import argparse
import os
import sys
import warnings

import numpy as np
import soundfile as sf
from tqdm.std import TqdmWarning
from pedalboard import Reverb
import pyloudnorm as pyln

import multiprocessing as mp
import queue

BUFFER_SIZE_SAMPLES = 1024 * 16
NOISE_FLOOR = 1e-4


def get_num_frames(f: sf.SoundFile) -> int:
    if len(f) > 2 ** 32:
        f.seek(0)
        last_position = f.tell()
        while True:
            f.seek(1024 * 1024 * 1024, sf.SEEK_CUR)
            new_position = f.tell()
            if new_position == last_position:
                f.seek(0)
                return new_position
            else:
                last_position = new_position
    else:
        return len(f)


def normalize_sound_file(f: sf.SoundFile, rate, isTP: bool, LU: float, TP: float):
    f.seek(0)
    data = f.read(-1)
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    sys.stderr.write(f"Measured loudness before normalization {loudness}...\n")
    if isTP:
        data_normalized = pyln.normalize.peak(data, TP)
    else:
        data_normalized = pyln.normalize.loudness(data, loudness, LU)
    f.seek(0)
    f.write(data_normalized)


def process_sound_file(input_queue, args):
    while True:
        try:
            file = input_queue.get_nowait()

        except queue.Empty:
            break
        else:
            args_input_file = file
            args_output_file = args_input_file + "_result.wav"
            reverb = Reverb()
            for arg in ('room_size', 'damping', 'wet_level', 'dry_level', 'width', 'freeze_mode'):
                setattr(reverb, arg, getattr(args, arg))

            with sf.SoundFile(args.input_directory + "/" + args_input_file) as input_file:
                sys.stderr.write(f"Writing to {args_output_file}...\n")
                if os.path.isfile(args.output_directory + "/" + args_output_file) and not args.overwrite:
                    raise ValueError(
                        f"Output file {args_output_file} already exists! (Pass -y to overwrite.)"
                    )
                with sf.SoundFile(
                        args.output_directory + "/" + args_output_file,
                        'w+',
                        samplerate=input_file.samplerate,
                        channels=input_file.channels
                ) as output_file:
                    length = get_num_frames(input_file)
                    length_seconds = length / input_file.samplerate

                    sys.stderr.write(f"Adding reverb to {length_seconds:.2f} seconds of audio...\n")
                    for dry_chunk in input_file.blocks(BUFFER_SIZE_SAMPLES, frames=length):
                        effected_chunk = reverb.process(
                            dry_chunk, sample_rate=input_file.samplerate, reset=False
                        )
                        output_file.write(effected_chunk)

                    if not args.cut_reverb_tail:
                        while True:
                            effected_chunk = reverb.process(
                                np.zeros((BUFFER_SIZE_SAMPLES, input_file.channels), np.float32),
                                sample_rate=input_file.samplerate,
                                reset=False,
                            )
                            if np.amax(np.abs(effected_chunk)) < NOISE_FLOOR:
                                break
                            output_file.write(effected_chunk)

                    output_file.flush()
                    normalize_sound_file(output_file, output_file.samplerate, args.isTP, args.LUFS, args.TP)
    return True


def main():
    warnings.filterwarnings("ignore", category=TqdmWarning)

    parser = argparse.ArgumentParser(description="Add reverb and normalize all audio files in given directory.")
    parser.add_argument("input_directory", help="The input directory to add reverb to.")
    parser.add_argument(
        "output_directory",
        help=(
            "The name of the output file to write to. If not provided, {input_file}.reverb.wav will"
            " be used."
        ),
        default=None,
    )

    reverb = Reverb()

    parser.add_argument("--room-size", type=float, default=reverb.room_size)
    parser.add_argument("--damping", type=float, default=reverb.damping)
    parser.add_argument("--wet-level", type=float, default=reverb.wet_level)
    parser.add_argument("--dry-level", type=float, default=reverb.dry_level)
    parser.add_argument("--width", type=float, default=reverb.width)
    parser.add_argument("--freeze-mode", type=float, default=reverb.freeze_mode)

    parser.add_argument(
        "--number_of_processes",
        help="Number of processes used for multiprocessing",
        default=8,
    )

    parser.add_argument(
        "--TP",
        help="Desirable True Peak",
        default=-1,
    )

    parser.add_argument(
        "--LUFS",
        help="Desirable LUFS",
        default=-12,
    )

    parser.add_argument(
        "--isTP",
        action="store_true",
        help=(
            "If passed, True Peak normalization is used."
            "LUFS otherwise."
        ),
    )

    parser.add_argument(
        "-y",
        "--overwrite",
        action="store_true",
        help="If passed, overwrite the output file if it already exists.",
    )

    parser.add_argument(
        "--cut-reverb-tail",
        action="store_true",
        help=(
            "If passed, remove the reverb tail to the end of the file. "
            "The output file will be identical in length to the input file."
        ),
    )
    args = parser.parse_args()

    input_directory = os.fsencode(args.input_directory)

    files = mp.Queue()
    total_number_of_processes = args.number_of_processes
    current_processes = []

    for file in os.listdir(input_directory):
        filename = os.fsdecode(file)
        files.put(filename)
        sys.stderr.write(f"Opening {filename}...\n")

    # manager = mp.Manager()
    # proxy = manager.Namespace()
    # proxy = args

    for i in range(total_number_of_processes):
        p = mp.Process(target=process_sound_file, args=(files, args))
        current_processes.append(p)
        p.start()

    for p in current_processes:
        p.join()

    sys.stderr.write("Done!\n")


if __name__ == "__main__":
    main()
