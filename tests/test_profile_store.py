"""
Phase 9B — tests for the in-memory enrollment profile store.

Covers save/get semantics, duplicate rejection, instance isolation, metadata
reporting, thread safety and the privacy guarantee that only aggregate
numeric summaries are ever stored (never raw events or identities).
"""

import threading

import pytest

from backend.app.services.profile_store import (
    InMemoryProfileStore,
    ProfileExistsError,
    StoredProfile,
    stored_fields,
    utc_now,
)

_FIELDS = {"user_ref", "centroid", "embedding_dim", "session_count", "created_at", "updated_at"}


def _profile(user_ref, n_sessions=2, dim=8):
    return StoredProfile(
        user_ref=user_ref,
        centroid=tuple(float(i) for i in range(dim)),
        embedding_dim=dim,
        session_count=n_sessions,
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-01-01T00:00:00+00:00",
    )


class TestSaveAndGet:
    def test_save_then_get_round_trip(self):
        store = InMemoryProfileStore()
        store.save(_profile("u1"))
        stored = store.get("u1")
        assert stored.user_ref == "u1"
        assert stored.centroid == tuple(range(8))
        assert stored.embedding_dim == 8
        assert stored.session_count == 2

    def test_get_unknown_returns_none(self):
        store = InMemoryProfileStore()
        assert store.get("nobody") is None

    def test_contains(self):
        store = InMemoryProfileStore()
        assert not store.contains("u1")
        store.save(_profile("u1"))
        assert store.contains("u1")

    def test_duplicate_save_raises_profile_exists(self):
        store = InMemoryProfileStore()
        store.save(_profile("u1"))
        with pytest.raises(ProfileExistsError) as info:
            store.save(_profile("u1"))
        assert info.value.code == "profile_exists"
        assert info.value.status_code == 409

    def test_duplicate_does_not_overwrite(self):
        store = InMemoryProfileStore()
        first = _profile("u1")
        store.save(first)
        with pytest.raises(ProfileExistsError):
            store.save(_profile("u1", n_sessions=9, dim=4))
        assert store.get("u1").session_count == 2
        assert store.get("u1").centroid == tuple(range(8))

    def test_save_rejects_non_stored_profile(self):
        store = InMemoryProfileStore()
        with pytest.raises(TypeError):
            store.save({"user_ref": "u1"})


class TestInstancesAndIntrospection:
    def test_instances_are_isolated(self):
        a = InMemoryProfileStore()
        b = InMemoryProfileStore()
        a.save(_profile("u1"))
        assert b.get("u1") is None
        assert a.count() == 1
        assert b.count() == 0

    def test_count_and_user_refs(self):
        store = InMemoryProfileStore()
        store.save(_profile("b"))
        store.save(_profile("a"))
        assert store.count() == 2
        assert store.user_refs() == ["a", "b"]

    def test_clear_removes_everything(self):
        store = InMemoryProfileStore()
        store.save(_profile("u1"))
        assert store.clear() == 1
        assert store.count() == 0
        assert store.get("u1") is None

    def test_list_profiles_excludes_centroid(self):
        store = InMemoryProfileStore()
        store.save(_profile("u1", n_sessions=3, dim=128))
        rows = store.list_profiles()
        assert rows == [
            {
                "user_ref": "u1",
                "embedding_dim": 128,
                "session_count": 3,
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2025-01-01T00:00:00+00:00",
            }
        ]
        assert "centroid" not in rows[0]


class TestPrivacyShape:
    def test_stored_fields_are_aggregate_only(self):
        assert set(stored_fields()) == set(_FIELDS)
        banned = {"keyboard_events", "mouse_events", "keyboard", "mouse", "events", "raw"}
        assert not (set(stored_fields()) & banned)

    def test_centroid_is_plain_float_tuple(self):
        stored = _profile("u1")
        assert isinstance(stored.centroid, tuple)
        assert all(isinstance(v, float) for v in stored.centroid)
        assert stored.session_count > 0
        assert stored.embedding_dim == len(stored.centroid)

    def test_utc_now_is_iso_string(self):
        assert isinstance(utc_now(), str)
        assert "T" in utc_now()


class TestThreadSafety:
    def test_concurrent_duplicate_save_one_wins(self):
        store = InMemoryProfileStore()
        barrier = threading.Barrier(2)
        results = []

        def attempt():
            barrier.wait()
            try:
                store.save(_profile("u1"))
                results.append("ok")
            except ProfileExistsError:
                results.append("exists")

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(results) == ["exists", "ok"]
        assert store.count() == 1

    def test_concurrent_distinct_saves_all_succeed(self):
        store = InMemoryProfileStore()

        def save(user_ref):
            store.save(_profile(user_ref))

        threads = [threading.Thread(target=save, args=("u{}".format(i),)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert store.count() == 8