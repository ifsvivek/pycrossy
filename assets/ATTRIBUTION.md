# Asset Attribution

The art, audio, font, and 3-D model assets bundled in this `assets/` directory were **not**
created for this project. They are reused from the open-source project below and are
included here under that project's licence.

## Original project

- **Project:** Expo Crossy Road (an open-source recreation of *Crossy Road*)
- **Author:** Evan Bacon and contributors
- **Repository:** https://github.com/EvanBacon/Expo-Crossy-Road
- **Licence:** MIT License

The low-poly voxel models were originally created by the project's author using the
open-source [MagicaVoxel](https://ephtracy.github.io/) editor.

## What is reused here

The following categories of assets were copied **unmodified** from the original
repository's `assets/` directory and are used at runtime by this Python port
(see [`../pycrossy/assets.py`](../pycrossy/assets.py)):

| Category | Count | Description |
| --- | --- | --- |
| 3-D models + textures | 74 | `.obj` meshes and `.png` textures for characters, vehicles, and environment tiles (`models/`) |
| Audio | 27 | `.wav` / `.mp3` sound effects — movement, deaths, vehicles, train, coins, UI (`audio/`) |
| Font | 1 | `retro.ttf` bitmap-style UI font (`fonts/`) |
| Title image | 1 | `title.png` "Crossy Road" logo (`images/`) |

Only the assets actually loaded by the game are bundled. Unused assets from the original
repository (app/launcher icons, hand/button UI sprites, unused audio variants, and the
full TypeScript source) were **not** copied. Assets that this project generates
procedurally at runtime (e.g. particle cubes, the procedural bezel) are not reused assets
and are not listed here.

A complete file-by-file inventory of every reused asset lives in
[`MANIFEST.txt`](MANIFEST.txt).

## Disclaimer (carried over from the original project)

The original project states it is *strictly for educational purposes only* and that its
author is *in no way associated with Hipster Whale* (the studio behind the commercial
*Crossy Road* game). This Python port is likewise an independent, non-commercial
reimplementation provided for educational and research use, and it retains the original
copyright and attribution as required by the MIT License.

## MIT License

The reused assets are distributed under the following licence. The copyright and
permission notice below is reproduced as required by the MIT License.

```
The MIT License (MIT)

Copyright (c) 2016-present Evan Bacon.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
