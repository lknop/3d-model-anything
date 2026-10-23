#!/usr/bin/env python3
"""Export upright JPEGs in exactly the order of a converted room's photos array.

Use with cmp_photo.sh and the viewer: photo N is NNNN.jpg, even when room filtering
selected non-contiguous original filenames. Source images stay in sensor orientation.
"""
import argparse
import json
from pathlib import Path
import tempfile
import shutil
from PIL import Image

# photo record `upright` → PIL transpose that turns the raw image upright
UPRIGHT = {'cw90': Image.Transpose.ROTATE_270, 'ccw90': Image.Transpose.ROTATE_90,
           '180': Image.Transpose.ROTATE_180, 'none': None}


def prepare(bundle, room, out, max_edge=1280):
    bundle = bundle.resolve()
    if out.exists():
        raise ValueError('output directory exists; choose a new directory')
    if max_edge < 1:
        raise ValueError('max-edge must be positive')
    photos = json.loads(room.read_text())['photos']
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp)
        for index, photo in enumerate(photos):
            source = (bundle / photo['file']).resolve()
            if not source.is_relative_to(bundle):
                raise ValueError('photo path escapes bundle')
            with Image.open(source) as raw:
                # Matches bundle_to_room's upright convention: cw90 (sensor -X = up) unless the
                # record says otherwise. Do not apply EXIF transpose as well: that would rotate twice.
                image = raw.convert('RGB')
                turn = UPRIGHT[photo.get('upright', 'cw90')]
                if turn is not None:
                    image = image.transpose(turn)
                image.thumbnail((max_edge, max_edge))
                image.save(stage / f'{index:04d}.jpg', quality=90)
        out.mkdir()
        for path in stage.iterdir():
            shutil.move(str(path), out / path.name)
    return len(photos)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('room', type=Path, help='converted room JSON, not raw RoomPlan JSON')
    parser.add_argument('out', type=Path, help='new upright-photo directory')
    parser.add_argument('--max-edge', type=int, default=1280)
    args = parser.parse_args()
    try:
        print(f'{prepare(args.bundle, args.room, args.out, args.max_edge)} photos → {args.out}')
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
