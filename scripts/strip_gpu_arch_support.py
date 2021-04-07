#!/usr/bin/env python
#===============================================================================
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

import os
import pathlib
import argparse
import subprocess
import re


def main():
    parser = argparse.ArgumentParser(
        description='remove all code specific to given architecture from oneDNN'
    )
    parser.add_argument('flags',
                        nargs='+',
                        help='flag(s) to strip from library')
    parser.add_argument('--unifdef',
                        default="unifdef",
                        help='path to unifdef executable')
    parser.add_argument('--onednn_root',
                        default=os.path.abspath("."),
                        help='path to onednn source')
    args = parser.parse_args()

    if not os.path.exists(args.onednn_root + "/src/gpu/"):
        print("invalid path to oneDNN source: {} ".format(args.onednn_root))
        exit(1)
    unifdef_flag = ["-U{}".format(i) for i in args.flags]
    architectures = [
        i[i.lower().find("xe_"):].lower() for i in args.flags
        if i.lower().find("xe_") != -1
    ]
    print("Stripping architectures: {}".format(architectures))
    print("unifdef flags: {} ".format(unifdef_flag))

    gpu_root = pathlib.Path(args.onednn_root + "/src/gpu")
    sycl_root = pathlib.Path(args.onednn_root + "/src/sycl")
    root_dirs = [x for x in gpu_root.iterdir()]
    root_dirs += [x for x in sycl_root.iterdir()]
    process_dirs(root_dirs, args, unifdef_flag, architectures)


def process_dirs(root, args, unifdef_flag, architectures):
    arch_patterns = ['\S*{}_\S*'.format(arch.lower()) for arch in architectures]
    for f in root:
        if f.is_dir():
            process_dirs([x for x in f.iterdir()], args, unifdef_flag,
                         architectures)
        elif not re.match('\S*\.cl', str(f)):
            r = subprocess.run([args.unifdef, "-m"] + unifdef_flag + [str(f)],
                               capture_output=True)
            print("unifdefed: {}".format(str(f)))
        else:
            print("skipped: {}".format(str(f)))
        for arch in arch_patterns:
            if re.match(arch, str(f)):
                r = subprocess.run(["rm", str(f)])
                print("deleted: {}".format(str(f)))
                break


if __name__ == "__main__":
    main()
    exit(0)
