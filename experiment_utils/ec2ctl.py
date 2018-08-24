#!/usr/bin/env python
import datetime
import json
import logging
import multiprocessing
import os
import re
import sys

import click

import boto3
from experiment_utils.utils import query_yes_no
import experiment_utils.config as config
from doodad.ec2.autoconfig import AUTOCONFIG
import numpy as np

DEBUG_LOGGING_MAP = {
    0: logging.CRITICAL,
    1: logging.WARNING,
    2: logging.INFO,
    3: logging.DEBUG
}


@click.group()
@click.option('--verbose', '-v',
              help="Sets the debug noise level, specify multiple times "
                   "for more verbosity.",
              type=click.IntRange(0, 3, clamp=True),
              count=True)
@click.pass_context
def cli(ctx, verbose):
    logger_handler = logging.StreamHandler(sys.stderr)
    logger_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(logger_handler)
    logging.getLogger().setLevel(DEBUG_LOGGING_MAP.get(verbose, logging.DEBUG))


REGIONS = [
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-south-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "eu-central-1",
    "eu-west-1",
    "sa-east-1",
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
]


def get_clients():
    regions = REGIONS
    clients = []
    for region in regions:
        client = boto3.client(
            "ec2",
            region_name=region,
            aws_access_key_id=AUTOCONFIG.aws_access_key(),
            aws_secret_access_key=AUTOCONFIG.aws_access_secret(),
        )
        client.region = region
        clients.append(client)
    return clients


def _collect_instances(region):
    client = boto3.client(
        "ec2",
        region_name=region,
        aws_access_key_id=AUTOCONFIG.aws_access_key(),
        aws_secret_access_key=AUTOCONFIG.aws_access_secret(),
    )
    print("Collecting instances in region", region)
    instances = [x['Instances'][0] for x in client.describe_instances(
        Filters=[
            {
                'Name': 'instance.group-name',
                'Values': [
                    AUTOCONFIG.aws_security_groups()[0],
                ]
            },
            {
                'Name': 'instance-state-name',
                'Values': [
                    'running'
                ]
            }
        ]
    )['Reservations']]
    for instance in instances:
        instance['Region'] = region
    return instances


def get_all_instances():
    with multiprocessing.Pool(10) as pool:
        all_instances = sum(pool.map(_collect_instances, REGIONS), [])
    return all_instances


def get_name_tag(instance):
    if 'Tags' in instance:
        try:
            tags = instance['Tags']
            name_tag = [t for t in tags if t['Key'] == 'Name'][0]
            return name_tag['Value']
        except IndexError:
            return None
    return None


def get_exp_prefix_tag(instance):
    if 'Tags' in instance:
        try:
            tags = instance['Tags']
            name_tag = [t for t in tags if t['Key'] == 'exp_prefix'][0]
            return name_tag['Value']
        except IndexError:
            return None
    return None


@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='verbose')
def jobs(verbose):
    jobs = []
    for instance in get_all_instances():
        name = get_name_tag(instance)
        if verbose:
            print("{}: \n\t region: {}, id: {}, state: {}".format(
                name, instance['Region'], instance['InstanceId'], instance['State']['Name']))
        if (name is not None) and (instance['State']['Name'] != 'terminated'):
            jobs.append(name)

    for job in sorted(jobs):
        print(job)


@cli.command()
@click.argument('job')
def ssh(job):
    for instance in get_all_instances():
        name = get_name_tag(instance)
        if name == job:
            ip_addr = instance['PublicIpAddress']
            key_path = AUTOCONFIG.aws_key_path(instance['Region'])
            command = " ".join([
                "ssh",
                "-oStrictHostKeyChecking=no",
                "-oConnectTimeout=10",
                "-i",
                key_path,
                "-t",
                "ubuntu@" + ip_addr,
            ])
            print(command)
            os.system(command)
            return
    print("Not found!")


# @cli.command()
# @click.argument('job')
# @click.option('--deterministic', '-d', default=False, help='run policy in deterministic mode')
# def sim_policy(job, deterministic):
#     for instance in get_all_instances():
#         name = get_name_tag(instance)
#         if name == job:
#             ip_addr = instance['PublicIpAddress']
#             exp_prefix = get_exp_prefix_tag(instance)
#             key_name = config.ALL_REGION_AWS_KEY_NAMES[instance['Region']]
#             key_path = os.path.join(config.PROJECT_PATH, "private/key_pairs", key_name + ".pem")
#
#             copy_command = [
#                 "ssh",
#                 "-oStrictHostKeyChecking=no",
#                 "-oConnectTimeout=10",
#                 "-i",
#                 key_path,
#                 "ubuntu@{ip}".format(ip=ip_addr),
#                 "cp {project_path}/data/local/{exp_prefix}/{job}/params.pkl /tmp/params.pkl".format(
#                     project_path=config.PROJECT_PATH,
#                     exp_prefix=exp_prefix,
#                     job=job
#
#                 )
#             ]
#             print(" ".join(copy_command))
#             subprocess.check_call(copy_command)
#
#             command = [
#                 "scp",
#                 "-oStrictHostKeyChecking=no",
#                 "-oConnectTimeout=10",
#                 "-i",
#                 key_path,
#                 "ubuntu@{ip}:/tmp/params.pkl".format(
#                     ip=ip_addr,
#                     project_path=config.PROJECT_PATH,
#                     exp_prefix=exp_prefix,
#                     job=job
#                 ),
#                 "/tmp/params.pkl"
#             ]
#             print(" ".join(command))
#             subprocess.check_call(command)
#             if "conopt" in job or "analogy" in job:
#                 script = "sandbox/rocky/analogy/scripts/sim_policy.py"
#             else:
#                 script = "scripts/sim_policy.py"
#             command = [
#                 "python",
#                 os.path.join(config.PROJECT_PATH, script),
#                 "/tmp/params.pkl"
#             ]
#             if deterministic:
#                 command += ["--deterministic"]
#             subprocess.check_call(command)
#             return
#     print("Not found!")


@cli.command()
@click.argument('pattern')
def kill_f(pattern):
    print("trying to kill the pattern: ", pattern)
    to_kill = []
    to_kill_ids = {}
    for instance in get_all_instances():
        name = get_name_tag(instance)
        if name is None or pattern in name:
            instance_id = instance['InstanceId']
            region = instance['Region']
            if name is None:
                if any([x['GroupName'] in AUTOCONFIG.aws_security_groups() for x in instance['SecurityGroups']]):
                    if query_yes_no(question="Kill instance {} without name in region {} (security groups {})?".format(
                            instance_id, region, [x['GroupName'] for x in instance['SecurityGroups']])):
                        name = instance_id
            if name:
                if region not in to_kill_ids:
                    to_kill_ids[region] = []
                to_kill_ids[region].append(instance_id)
                to_kill.append(name)

    print("This will kill the following jobs:")
    print(", ".join(sorted(to_kill)))
    if query_yes_no(question="Proceed?", default="no"):
        for client in get_clients():
            print("Terminating instances in region", client.region)
            ids = to_kill_ids.get(client.region, [])
            if len(ids) > 0:
                client.terminate_instances(
                    InstanceIds=to_kill_ids.get(client.region, [])
                )


@cli.command()
@click.argument('job')
def kill(job):
    to_kill = []
    to_kill_ids = {}
    for instance in get_all_instances():
        name = get_name_tag(instance)
        if name == job:
            region = instance['Region']
            if region not in to_kill_ids:
                to_kill_ids[region] = []
            to_kill_ids[region].append(instance['InstanceId'])
            to_kill.append(name)
            break

    print("This will kill the following jobs:")
    print(", ".join(sorted(to_kill)))
    if query_yes_no(question="Proceed?", default="no"):
        for client in get_clients():
            print("Terminating instances in region", client.region)
            ids = to_kill_ids.get(client.region, [])
            if len(ids) > 0:
                client.terminate_instances(
                    InstanceIds=to_kill_ids.get(client.region, [])
                )


def fetch_zone_prices(instance_type, zone, duration):
    clients = get_clients()
    for client in clients:
        if zone.startswith(client.region):

            all_prices = []
            all_ts = []
            for response in client.get_paginator('describe_spot_price_history').paginate(
                    InstanceTypes=[instance_type],
                    ProductDescriptions=['Linux/UNIX'],
                    AvailabilityZone=zone,
            ):
                history = response['SpotPriceHistory']
                prices = [float(x['SpotPrice']) for x in history]
                timestamps = [x['Timestamp'] for x in history]

                all_prices.extend(prices)
                all_ts.extend(timestamps)

                if len(all_ts) > 0:

                    delta = max(all_ts) - min(all_ts)
                    if delta.total_seconds() >= duration:
                        break

            return zone, all_prices, all_ts


def fetch_zones(region):
    clients = get_clients()
    for client in clients:
        if client.region == region:
            zones = [x['ZoneName'] for x in client.describe_availability_zones()['AvailabilityZones']]
            return zones


@cli.command()
@click.argument('instance_type')
@click.option('--duration', '-d', help="Specify the duration to measure the maximum price. Defaults to 1 day. "
                                       "Examples: 100s, 1h, 2d, 1w", type=str, default='1d')
def spot_history(instance_type, duration):
    num_duration = int(duration[:-1])
    if re.match(r"^(\d+)d$", duration):
        duration = int(duration[:-1]) * 86400
        print("Querying maximum spot price in each zone within the past {duration} day(s)...".format(
            duration=num_duration))
    elif re.match(r"^(\d+)h$", duration):
        duration = int(duration[:-1]) * 3600
        print("Querying maximum spot price in each zone within the past {duration} hour(s)...".format(
            duration=num_duration))
    elif re.match(r"^(\d+)w$", duration):
        duration = int(duration[:-1]) * 86400 * 7
        print("Querying maximum spot price in each zone within the past {duration} week(s)...".format(
            duration=num_duration))
    elif re.match(r"^(\d+)m$", duration):
        duration = int(duration[:-1]) * 86400 * 30
        print("Querying maximum spot price in each zone within the past {duration} month(s)...".format(
            duration=num_duration))
    elif re.match(r"^(\d+)s$", duration):
        duration = int(duration[:-1])
        print("Querying maximum spot price in each zone within the past {duration} second(s)...".format(
            duration=num_duration))
    else:
        raise ValueError("Unrecognized duration: {duration}".format(duration))

    with multiprocessing.Pool(100) as pool:
        print('Fetching the list of all availability zones...')
        zones = sum(pool.starmap(fetch_zones, [(x,) for x in REGIONS]), [])
        print('Querying spot price in each zone...')
        results = pool.starmap(fetch_zone_prices, [(instance_type, zone, duration) for zone in zones])

        price_list = []

        for zone, prices, timestamps in results:
            if len(prices) > 0:
                sorted_prices_ts = sorted(zip(prices, timestamps), key=lambda x: x[1])
                cur_time = datetime.datetime.now(tz=sorted_prices_ts[0][1].tzinfo)
                sorted_prices, sorted_ts = [np.asarray(x) for x in zip(*sorted_prices_ts)]
                cutoff = cur_time - datetime.timedelta(seconds=duration)

                valid_ids = np.where(np.asarray(sorted_ts) > cutoff)[0]
                if len(valid_ids) == 0:
                    first_idx = 0
                else:
                    first_idx = max(0, valid_ids[0] - 1)

                max_price = max(sorted_prices[first_idx:])

                price_list.append((zone, max_price))

        print("Spot pricing information for instance type {type}".format(type=instance_type))

        list_string = ''
        for zone, price in sorted(price_list, key=lambda x: x[1]):
            print("Zone: {zone}, Max Price: {price}".format(zone=zone, price=price))
            list_string += "'{}', ".format(zone)
        print(list_string)


@cli.command()
def ami():
    clients = get_clients()
    for client in clients:
        images = client.describe_images(Owners=['self'])['Images']
        for img in images:
            print('{name} in {region}'.format(name=img['Name'], region=client.region))


if __name__ == '__main__':
    cli()
