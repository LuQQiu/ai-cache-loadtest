#!/usr/bin/python

# Require python > 3.0

import alluxio
import argparse
import csv
import os
import torch
import time
import sys

from prometheus_client import multiprocess, CollectorRegistry, Summary
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group
from urllib.parse import urlparse

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)
LATENCY = Summary("read_latency", "Read request latency")


if sys.version_info.major < 3:
    raise RuntimeError("python3 is required")


def read_filelist(csv_file, max_len):
    with open(csv_file) as cf:
        reader = csv.reader(cf)
        urls = []
        count = 0
        for row in reader:
            urls.append(row[0])
            count += 1
            if count >= max_len:
                break
    print("read data, number of files:", len(urls))
    return urls


class AlluxioWorkerDataset(Dataset):
    def __init__(self, size, filelist, prefix):
        self.size = size
        self.url = read_filelist(filelist, size)
        self.size = len(self.url)
        re = urlparse(prefix)
        host = re.hostname
        port = re.port
        self.client = alluxio.Client(host, port)
        self.prefix = re.path

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        file_path = self.prefix + self.url[index]
        v = 0
        try:
            start = time.time()
            with self.client.open(file_path, 'r') as f:
                data = f.read()
            v = len(data)
            LATENCY.observe(time.time() - start)
        except Exception as e:
            print(e)
            pass
        return v


def start_load(args):
    batch_size = 1024
    torch.distributed.init_process_group("gloo")
    mydataset = AlluxioWorkerDataset(size=args.number_of_files, filelist=args.inputfile, prefix=args.path_prefix)
    sampler = DistributedSampler(mydataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
            mydataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=args.workers,
            drop_last=True,
            persistent_workers=True)

    count = 0
    st = time.time()
    for v in data_loader:
        if count % 100 == 0 and args.local_rank == 0:
           print("processing ", count)
        count += 1
    total_ts = time.time() - st;
    print("time cost for {} items:".format(count), total_ts)
    if count > 0:
        print("Overall Avg.Latency {:.2f} ms".format(total_ts * 1000.0 / count / batch_size))
        for metric in registry.collect():
            name = metric.name
            num = 0
            total = 0
            for item in metric.samples:
                if item.name == name + "_sum":
                    total = item.value
                elif item.name == name + "_count":
                    num = item.value
            print("{}: {:.2f} ms".format(name, total * 1000.0/num))


def main():
    parser = argparse.ArgumentParser(description='Alluxio BenchTest')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('-i', '--inputfile', default='ossdata.csv', type=str,
                        help='path to input file list')
    parser.add_argument('-n', '--number_of_files', default=5000000, type=int, metavar='N',
                        help='number of files to be processed')
    parser.add_argument('-P', '--path_prefix', type=str, metavar='N',
                        help='path prefix of the list files')
    parser.add_argument('-r', '--local_rank', type=int, metavar='N',
                        help='local rank')
    args = parser.parse_args()
    start_load(args)


if __name__ == "__main__":
    main()
