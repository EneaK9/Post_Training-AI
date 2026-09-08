import numpy as np

from outlier_ai.core.embeddings import HashEmbedder, cosine_distance


def test_hash_embedder_is_deterministic_and_normalized():
    e = HashEmbedder(64)
    a = e.embed(["objection flip on a price objection"])
    b = HashEmbedder(64).embed(["objection flip on a price objection"])
    assert a.shape == (1, 64)
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-5


def test_similar_texts_are_closer():
    e = HashEmbedder(384)
    v = e.embed(
        [
            "cards: contrarian + specific-numbers\n"
            "angle: everything you know about skincare is wrong",
            "cards: contrarian + specific-numbers\n"
            "angle: everything you were told about skincare is wrong",
            "cards: creativity-and-humor + meme-format\nangle: my wallet left the chat",
        ]
    )
    assert cosine_distance(v[0], v[1]) < cosine_distance(v[0], v[2])
    assert cosine_distance(v[0], v[0]) < 1e-6


def test_cosine_distance_handles_zero_vectors():
    assert cosine_distance([0, 0], [1, 0]) == 1.0
    assert HashEmbedder(16).embed([""]).sum() == 0
