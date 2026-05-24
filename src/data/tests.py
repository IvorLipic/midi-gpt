from src.data.gigamidi_loop_filter import debug_check_identical_loop

if __name__ == "__main__":
    # True
    debug_check_identical_loop("data/GigaMIDI/extracted_loops_v1/train/4-4/00003093598b6e48be9d674f97dae7d3_loop5_track1_178560.mid")
    # True
    debug_check_identical_loop("data/GigaMIDI/extracted_loops_v1/train/4-4/00003b6e5616bc9ac512423c8be241d9_loop4_track1_130560.mid")
    # False
    debug_check_identical_loop("data/GigaMIDI/extracted_loops_v1/train/4-4/0000b1481c41e1270fccc5d1c38aeea0_loop1_track0_1536.mid")