#!/usr/bin/env python3

# Author: Eliska Kloberdanz

import sys
import os
import urllib.request
from bs4 import BeautifulSoup

GOAL = ' '

class Node:
    def __init__(self, name, path):
        self.name = name
        self.path = path


def get_links(filename):
    s = read_page(filename)
    soup = BeautifulSoup(s, 'lxml')
    links = soup.findAll('a', href=True)
    l = []
    for link in links:
        href = link['href'].strip('/')
        text = link.text
        l.append((href, text))
    return l


def read_page(page):
    print('openening: {}'.format(page))
    try:
        response = urllib.request.urlopen(page)
        s = response.read()
        return s.decode()
    except urllib.error.HTTPError as e:
        return ''


def get_num_query_words(s):
    global GOAL
    count = 0
    for word in s.split():
        if word in GOAL:
            count += 1
    return count


def breadth_first_search(start, GOAL, directory):
    if GOAL in read_page(start).lower():
        return [start], set()
    search_space = [Node(start, [start])]
    processed = set()
    while search_space:
        node = search_space.pop(0)
        next_links = [os.path.join(directory, link)
                      for (link, _) in get_links(node.name)
                      if link not in processed]
        for next_node in next_links:
            if GOAL in read_page(next_node).lower():
                return node.path + [next_node], processed
            elif next_node not in processed:
                processed.add(next_node)
                new_path = node.path + [next_node]
                search_space.append(Node(next_node, new_path))



def best_first_search(start, goal, directory):
    if goal in read_page(start):
        return [start], set()
    search_space = [Node(start, [start])]
    processed = set()
    while search_space:
        node = search_space.pop(0)

        next_links = []
        i = 0
        for (link, text) in get_links(node.name):
            if link not in processed:
                full_path = os.path.join(directory, link)
                next_links.append((full_path, get_num_query_words(text)))

        next_links.sort(key=lambda x: x[1], reverse=True)
        for (next_node, _) in next_links:
            if goal in read_page(next_node):
                return node.path + [next_node], processed
            elif next_node not in processed:
                processed.add(next_node)
                new_path = node.path + [next_node]
                search_space.append(Node(next_node, new_path))


def beam_search(start, GOAL, directory):
    contents = read_page(start)
    if GOAL in contents.lower():
        return [start], set()
    search_space = [Node(start, [start])]
    processed = set()
    while search_space:
        node = search_space.pop(0)

        next_links = []
        for (link, text) in get_links(node.name):
            if link not in processed:
                full_path = os.path.join(directory, link)
                next_links.append((full_path, get_num_query_words(text)))

        next_links.sort(key=lambda x: x[1], reverse=True)
        for (next_node, _) in next_links[:20]:
            if GOAL in read_page(next_node).lower():
                return node.path + [next_node], processed
            elif next_node not in processed:
                processed.add(next_node)
                new_path = node.path + [next_node]
                search_space.append(Node(next_node, new_path))


def _depth_first_search(node, GOAL, directory, processed, path):
    if node in processed:
        return None, None

    processed.add(node)
    if GOAL in read_page(node).lower():
        return path + [node], processed

    next_links = [os.path.join(directory, link)
                  for link in [x[0] for x in get_links(node)]
                  if link not in processed]

    for link in next_links:
        (ret, _) = _depth_first_search(link, GOAL, directory, processed, path + [node])
        if ret is not None:
            return ret, processed
    return None, None


def depth_first_search(start, GOAL, directory):
    processed = set()
    path = []
    return _depth_first_search(start, GOAL, directory, processed, path)


def print_results(path, visited):
    print('\n' 'number of nodes to GOAL: ' + str(len(path)) + '\n' 'path to GOAL:')
    print('\n'.join(path))
    #print('\n' 'number of nodes visited: ' + str(len(visited)) + '\n' 'nodes visited:')
    #print('\n'.join(visited))
    print()


def main():
    start = sys.argv[1]
    search_strategy = sys.argv[2]
    GOAL = ' '.join(sys.argv[3:])
    print('searching for:'+ str(GOAL))

    if search_strategy == 'beam':
        path, visited = beam_search(start, GOAL, start)
        print('Beam Search')
        print_results(path, visited)

    elif search_strategy == 'best':
        path, visited = best_first_search(start, GOAL, start)
        print('Beam Search')
        print_results(path, visited)

    elif search_strategy == 'depth':
        path, visited = depth_first_search(start, GOAL, start)
        print('Depth First Search')
        print_results(path, visited)

    elif search_strategy == 'breadth':
        path, visited = breadth_first_search(start, GOAL, start)
        print('Breadth First Search')
        print_results(path, visited)

    else:
        print('Invalid arguments')
        print('usage:\npython3 {} <start> <search_strategy>'.format(sys.argv[0]))
        print('search_strategy can be beam or depth or breadth or best')


if __name__ == '__main__':
    main()
    print('done')
