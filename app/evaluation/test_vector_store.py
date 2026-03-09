import os
import shutil
import numpy as np
from vector_store.active_segment_manager import VectorStoreManager


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def clean_segments():
    if os.path.exists("segments"):
        shutil.rmtree("segments")


def count_disk_segments():
    if not os.path.exists("segments"):
        return 0
    return len([f for f in os.listdir("segments") if f.endswith(".faiss")])


def assert_active_bound(manager, max_size):
    assert manager.active_index.ntotal <= max_size, \
        f"Active segment exceeded limit: {manager.active_index.ntotal}"


def assert_no_cross_user(I, user_id):
    for vid in I[0]:
        if vid != -1:
            assert (vid >> 32) == user_id, \
                f"Cross-user leakage detected: {vid}"


# --------------------------------------------------
# TEST CONFIG
# --------------------------------------------------

clean_segments()

dim = 128
upgrade_threshold = 50
max_segment_size = 100

manager = VectorStoreManager(
    dim=dim,
    segment_dir="segments",
    upgrade_threshold=upgrade_threshold,
    max_vectors_per_segment=max_segment_size,
    nlist=4
)

# --------------------------------------------------
# TEST 1: Flat exact search
# --------------------------------------------------

print("\n--- TEST 1: Flat Exact Search ---")

vectors_u1 = np.random.rand(40, dim).astype(np.float32)
ids_u1 = np.array([(1 << 32) | i for i in range(40)], dtype=np.int64)

manager.add_vectors(vectors_u1, ids_u1)

D, I = manager.search(vectors_u1[0:1], user_id=1, k=5)
assert I[0][0] == ids_u1[0]
assert_active_bound(manager, max_segment_size)
print("Flat search OK")


# --------------------------------------------------
# TEST 2: Upgrade preserves data
# --------------------------------------------------

print("\n--- TEST 2: Upgrade Integrity ---")

vectors_more = np.random.rand(20, dim).astype(np.float32)
ids_more = np.array([(1 << 32) | (i + 40) for i in range(20)], dtype=np.int64)

manager.add_vectors(vectors_more, ids_more)

assert manager.active_type == "ivf"
assert manager.active_index.ntotal == 60

D, I = manager.search(vectors_u1[0:1], user_id=1, k=5)
assert I[0][0] == ids_u1[0]

print("Upgrade preserved vectors")


# --------------------------------------------------
# TEST 3: Large batch splitting
# --------------------------------------------------

print("\n--- TEST 3: Large Batch Splitting ---")

vectors_large = np.random.rand(250, dim).astype(np.float32)
ids_large = np.array([(1 << 32) | (i + 100) for i in range(250)], dtype=np.int64)

manager.add_vectors(vectors_large, ids_large)

# Expected:
# Previously had 60
# After splitting and rotations:
# 100 + 100 + remainder in active

disk_count = count_disk_segments()
print("Disk segments:", disk_count)

assert disk_count >= 2
assert_active_bound(manager, max_segment_size)

# Check random sample search
D, I = manager.search(vectors_large[10:11], user_id=1, k=3)
assert I[0][0] == ids_large[10]

print("Batch splitting correct")


# --------------------------------------------------
# TEST 4: Multi-user strict isolation
# --------------------------------------------------

print("\n--- TEST 4: Multi-User Isolation ---")

vectors_u2 = np.random.rand(150, dim).astype(np.float32)
ids_u2 = np.array([(2 << 32) | i for i in range(150)], dtype=np.int64)

manager.add_vectors(vectors_u2, ids_u2)

D, I = manager.search(vectors_u2[5:6], user_id=2, k=5)
assert_no_cross_user(I, 2)

# Ensure user 1 query does not leak user 2
D, I = manager.search(vectors_u1[0:1], user_id=1, k=5)
assert_no_cross_user(I, 1)

print("User isolation strict")


# --------------------------------------------------
# TEST 5: No Data Loss Across Segments
# --------------------------------------------------

print("\n--- TEST 5: Data Completeness ---")

# Sample random vectors from all users
test_vectors = [
    (vectors_u1[10:11], ids_u1[10]),
    (vectors_large[50:51], ids_large[50]),
    (vectors_u2[20:21], ids_u2[20])
]

for vec, expected_id in test_vectors:
    uid = expected_id >> 32
    D, I = manager.search(vec, user_id=uid, k=5)
    assert expected_id in I[0], f"Missing vector {expected_id}"

print("No data loss across rotations")


# --------------------------------------------------
# TEST 6: Restart Recovery Integrity
# --------------------------------------------------

print("\n--- TEST 6: Restart Recovery ---")

manager = VectorStoreManager(
    dim=dim,
    segment_dir="segments",
    upgrade_threshold=upgrade_threshold,
    max_vectors_per_segment=max_segment_size,
    nlist=4
)

# Re-run sample checks
for vec, expected_id in test_vectors:
    uid = expected_id >> 32
    D, I = manager.search(vec, user_id=uid, k=5)
    assert expected_id in I[0], f"Restart lost vector {expected_id}"

print("Restart recovery integrity OK")


# --------------------------------------------------
# TEST 7: Segment Size Invariants
# --------------------------------------------------

print("\n--- TEST 7: Segment Size Invariants ---")

for seg in manager.mapped_segments:
    assert seg.ntotal <= max_segment_size, \
        f"Disk segment exceeded size: {seg.ntotal}"

assert manager.active_index.ntotal <= max_segment_size

print("Segment size invariants respected")


print("\nALL THOROUGH TESTS PASSED.")
