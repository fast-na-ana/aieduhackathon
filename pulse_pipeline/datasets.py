"""
Lists the training videos with their labels and identities.

subject        = whose body (and so whose pulse) is on camera. For a face swap that is the DRIVER
                 video's person, not the face that was pasted on.
source_subject = whose face was pasted on (= subject for real videos).
pair           = a video aligned frame by frame with this one (a fake's original, or one of a real
                 video's fakes). The difference between the two shows which face was swapped, so
                 the tracker follows that face when several people are in shot.

DFDC  metadata.json gives each fake's original video. DFDC has no person IDs, and the same actor
      appears in several originals, so dfdc_people.py clusters the originals by face into
      people and saves dfdc_people.json (original stem -> person id). A fake's subject is its
      original's person. Without that file, the original video stands in for the person and a
      subject split of DFDC can leak a person across splits. The pasted face's identity is not
      published, so source_subject is a unique "?..." placeholder.
DFD   real:  <actor>__<scene>.mp4
      fake:  <driver>_<face>__<scene>__<hash>.mp4. The driver is the FIRST id: checked on 24
      random fakes, whose backgrounds match the first id's original to <1 grey level on average.
      The two walking scenes are the same footage filed under both partners, and the tracker
      follows the same face in both copies. dfd_fixes.json lists the clips to leave out ("drop")
      and those whose tracked face is another actor ("subject"), each with the reason.
"""
import glob
import json
import os
from collections import defaultdict

DATA_ROOT = r"D:\DeepFake_Data"
DFDC_DIR = os.path.join(DATA_ROOT, "dfdc", "dfdc_train_part_00", "dfdc_train_part_0")
DFD_REAL_DIR = os.path.join(DATA_ROOT, "dfd", "DFD_original sequences")
DFD_FAKE_DIR = os.path.join(DATA_ROOT, "dfd", "DFD_manipulated_sequences", "DFD_manipulated_sequences")
DFDC_PEOPLE = os.path.join(os.path.dirname(__file__), "dfdc_people.json")
DFD_FIXES = os.path.join(os.path.dirname(__file__), "dfd_fixes.json")


def _json_or_empty(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _clip(dataset, path, label, subject, source, scene, pair):
    stem = os.path.splitext(os.path.basename(path))[0]
    return dict(clip=f"{dataset}/{stem}", dataset=dataset, path=path, label=label,
                attack="swap" if label else "real", subject=subject, source_subject=source,
                scene=scene, pair=pair)


def dfdc_clips(folder=DFDC_DIR, people_path=DFDC_PEOPLE):
    with open(os.path.join(folder, "metadata.json")) as f:
        meta = json.load(f)
    people = _json_or_empty(people_path)
    person = lambda stem: f"dfdc/{people.get(stem, stem)}"
    fakes_of = defaultdict(list)
    for name, m in sorted(meta.items()):
        if m["label"] == "FAKE":
            fakes_of[m["original"]].append(name)
    clips = []
    for name, m in sorted(meta.items()):
        path = os.path.join(folder, name)
        stem = os.path.splitext(name)[0]
        if m["label"] == "FAKE":
            orig = os.path.splitext(m["original"])[0]
            clips.append(_clip("dfdc", path, 1, person(orig), f"?dfdc/{stem}", "dfdc",
                               os.path.join(folder, m["original"])))
        else:
            pair = os.path.join(folder, fakes_of[name][0]) if fakes_of[name] else ""
            clips.append(_clip("dfdc", path, 0, person(stem), person(stem), "dfdc", pair))
    return clips


def dfd_clips(real_dir=DFD_REAL_DIR, fake_dir=DFD_FAKE_DIR, fixes_path=DFD_FIXES):
    fixes = _json_or_empty(fixes_path)
    drop, actual = fixes.get("drop", {}), fixes.get("subject", {})
    fakes = sorted(glob.glob(os.path.join(fake_dir, "*.mp4")))
    fakes_of = defaultdict(list)
    clips = []
    for path in fakes:
        stem = os.path.basename(path)[:-4]
        if stem in drop:
            continue
        ids, scene, _ = stem.split("__")
        driver, face = ids.split("_")
        orig = os.path.join(real_dir, f"{driver}__{scene}.mp4")
        fakes_of[orig].append(path)
        clips.append(_clip("dfd", path, 1, f"dfd/{actual.get(stem, driver)}", f"dfd/{face}", scene,
                           orig if os.path.exists(orig) else ""))
    for path in sorted(glob.glob(os.path.join(real_dir, "*.mp4"))):
        stem = os.path.basename(path)[:-4]
        if stem in drop:
            continue
        actor, scene = stem.split("__")
        actor = actual.get(stem, actor)
        pair = fakes_of[path][0] if fakes_of[path] else ""
        clips.append(_clip("dfd", path, 0, f"dfd/{actor}", f"dfd/{actor}", scene, pair))
    return clips


def all_clips():
    return dfdc_clips() + dfd_clips()
