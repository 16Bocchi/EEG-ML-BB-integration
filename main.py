import time

from modules import binaural_beat_generator as bbg


def main():
    # Create the binaural beat generator
    gen = bbg.BinauralBeatGenerator(
        sample_rate=44100,
        carrier_freq=440,  # A4 note
        beat_freq=10,  # 10 Hz beat frequency
        waveform="sine",
        amplitude=0.2,
        ramp_duration=2,  # ramp changes over 2 seconds
    )

    # Start audio playback
    gen.start()
    print("Playing binaural beats...")

    # Play initial tone for 5 seconds
    time.sleep(5)

    # Change frequencies smoothly (over 2 seconds ramp)
    print("Changing frequencies to carrier=480Hz, beat=7Hz")
    gen.update_frequencies(new_carrier_freq=480, new_beat_freq=7)

    # Change waveform after 5 seconds
    time.sleep(5)
    print("Changing waveform to 'square'")
    gen.update_waveform("square")

    # Change waveform after 5 seconds
    time.sleep(5)
    print("Changing waveform to 'saw'")
    gen.update_waveform("saw")

    # Let it play for 5 more seconds
    time.sleep(5)

    # Stop playback
    gen.stop()
    print("Stopped playback.")


if __name__ == "__main__":
    main()
