import os
import threading
import numpy as np
import faiss


class VectorStoreManager:

    def __init__(
        self,
        dim: int,
        segment_dir: str,
        upgrade_threshold: int = 5000,
        max_vectors_per_segment: int = 100_000,
        nlist: int = 32,
        logger=None
    ):
        self.dim = dim
        self.segment_dir = segment_dir
        self.upgrade_threshold = upgrade_threshold
        self.max_vectors_per_segment = max_vectors_per_segment
        self.nlist = nlist
        self.logger = logger

        os.makedirs(segment_dir, exist_ok=True)

        self._lock = threading.Lock()
        self.active_index = self._create_flat_index()
        self.active_type = "flat"

        self.mapped_segments = []

        self._load_existing_segments()

    def _create_flat_index(self):
        base = faiss.IndexFlatL2(self.dim)
        return faiss.IndexIDMap(base)

    def _create_ivf_index(self):
        quantizer = faiss.IndexFlatL2(self.dim)
        ivf = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist)
        return faiss.IndexIDMap(ivf)


    def add_vectors(self, vectors: np.ndarray, ids: np.ndarray):
        ids=np.asarray(ids).astype(np.int64)
        vectors=np.asarray(vectors).astype(np.float32)

        if len(vectors) == 0:
            return True
        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  
        vectors = vectors / norms

        with self._lock:
            try:
                # Upgrade if still flat
                if self.active_type == "flat":
                    try:
                        self.active_index.add_with_ids(vectors, ids)
                        if self.active_index.ntotal >= self.upgrade_threshold:
                            self._upgrade_flat_to_ivf()
                        self.logger.info(f'[VECTOR STORE] SUCCESS: {len(vectors)} vectors added')
                        return True
                    except Exception as e:
                        self.logger.exception(f"[VECTOR STORE] ADD ERROR: {e}")
                        return False


                # IVF mode
                remaining_vectors = vectors
                remaining_ids = ids
                # print(len(remaining_vectors))
                while len(remaining_vectors) > 0:

                    space_left = self.max_vectors_per_segment - self.active_index.ntotal

                    # If full, rotate
                    if space_left == 0:
                        self._rotate_ivf_segment()
                        continue

                    take = min(space_left, len(remaining_vectors))

                    self.active_index.add_with_ids(
                        remaining_vectors[:take],
                        remaining_ids[:take].astype(np.int64)
                    )

                    remaining_vectors = remaining_vectors[take:]
                    remaining_ids = remaining_ids[take:]
                self.logger.info(f'[VECTOR STORE] SUCCESS: {len(vectors)} vectors added')
                return True

            except Exception as e:
                if self.logger:
                    self.logger.exception(f"[VECTOR STORE] ADD ERROR: {e}")
                return False

    def _upgrade_flat_to_ivf(self):

        if self.logger:
            self.logger.info("[VECTOR STORE] Upgrading Flat -> IVF")

        flat_index = self.active_index
        ntotal = flat_index.ntotal

        xb = flat_index.index.reconstruct_n(0, ntotal)
        ids = faiss.vector_to_array(flat_index.id_map)
        ivf_index = self._create_ivf_index()
        ivf_index.train(xb)
        ivf_index.add_with_ids(xb, ids)
        self.active_index = ivf_index
        self.active_type = "ivf"
    def _rotate_ivf_segment(self):
        old_index = self.active_index
        if old_index.ntotal == 0:
            print('No vectors to rotate')
            return
        segment_id = len(self.mapped_segments) + 1
        path = os.path.join(
            self.segment_dir,
            f"segment_{segment_id}.faiss"
        )
        faiss.write_index(old_index, path)
        print(path)
        mapped = faiss.read_index(
            path
            # faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
        )
        
        print('Hello here')
        self.mapped_segments.append(mapped)
        print('Current length of mapped segments:',len(self.mapped_segments))
        new_ivf = self._create_ivf_index()
        training_data = self._sample_training_data()

        if training_data is not None:
            new_ivf.train(training_data)

        self.active_index = new_ivf
    # def _rotate_ivf_segment(self):
    #     old_index = self.active_index
    #     if old_index.ntotal == 0:
    #         return

    #     segment_id = len(self.mapped_segments) + 1
    #     path = os.path.join(self.segment_dir, f"segment_{segment_id}.faiss")
    #     ivfdata_path = os.path.join(self.segment_dir, f"segment_{segment_id}.ivfdata")

    #     # --- MMAP CONVERSION MAGIC ---
    #     # Extract the underlying IVF index from the IDMap
    #     ivf_base = faiss.downcast_index(old_index.index)
        
        
    #     if isinstance(ivf_base, faiss.IndexIVF):
    #         # Create on-disk inverted lists backing file
    #         print(ivfdata_path)
    #         invlists = faiss.OnDiskInvertedLists(
    #             int(ivf_base.nlist), int(ivf_base.code_size), ivfdata_path
    #         )
    #         print('hiii')
    #         # Replace in-memory array with disk array
    #         ivf_base.replace_invlists(invlists)

    #     # Now write the index (it will link to the .ivfdata file)
    #     faiss.write_index(old_index, path)
        
    #     # You can now safely uncomment the MMAP flags!
    #     print('Hi')
    #     mapped = faiss.read_index(
    #         path,
    #         faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
    #     )
    #     print('Hello')
    #     self.mapped_segments.append(mapped)

    #     # --- SETUP NEW ACTIVE INDEX ---
    #     new_ivf = self._create_ivf_index()
    #     training_data = self._sample_training_data()

    #     # Bug Fix: Ensure we actually trained before assignment
    #     if training_data is not None and len(training_data) >= self.nlist:
    #         new_ivf.train(training_data)
    #         self.active_index = new_ivf
    #     else:
    #         # Fallback to a flat index if we couldn't get training data
    #         self.active_index = self._create_flat_index()
    #         self.active_type = "flat"
    def _sample_training_data(self, sample_size=5000):

        vectors = []

        for seg in self.mapped_segments:
            try:
                ntotal = seg.ntotal
                if ntotal == 0:
                    continue

                take = min(sample_size, ntotal)
                xb = seg.index.reconstruct_n(0, take)
                vectors.append(xb)

            except Exception:
                continue

        if not vectors:
            return None

        return np.vstack(vectors)

    def _load_existing_segments(self):
        for fname in sorted(os.listdir(self.segment_dir)):
            if fname.endswith(".faiss"):
                path = os.path.join(self.segment_dir, fname)
                idx = faiss.read_index(
                    path
                    # faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
                )
                self.mapped_segments.append(idx)
    def search(self, query_vector: np.ndarray, user_id: int, k: int):

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        start_id = user_id << 32
        end_id = (user_id + 1) << 32
        print(start_id,end_id)
        selector = faiss.IDSelectorRange(int(start_id), int(end_id))

        all_D = []
        all_I = []

        if self.active_type == "ivf":

            params = (faiss.SearchParametersIVF(sel=selector))

            D, I = self.active_index.search(
                query_vector,
                k,
                params=params
            )

        else:
            D, I = self.active_index.search(query_vector, k * 5)
            mask = [
                (vid >= start_id and vid < end_id)
                for vid in I[0]
            ]

            I = I[:, mask]
            D = D[:, mask]

        all_D.append(D)
        all_I.append(I)
        for seg in self.mapped_segments:

            if True:

                params = (faiss.SearchParametersIVF(sel=selector))
                # params.sel = selector

                D_seg, I_seg = seg.search(
                    query_vector,
                    k,
                    params=params
                )
            else:
                D_seg, I_seg = seg.search(query_vector, k)

            all_D.append(D_seg)
            all_I.append(I_seg)

        if not all_D:
            return None, None
        D_all = np.hstack(all_D)
        I_all = np.hstack(all_I)
        top_idx = np.argsort(D_all, axis=1)[:, :k]
        final_D = np.take_along_axis(D_all, top_idx, axis=1)
        final_I = np.take_along_axis(I_all, top_idx, axis=1)

        return final_D, final_I