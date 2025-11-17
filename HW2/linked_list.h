
#ifndef _LINKED_LIST_H_
#define _LINKED_LIST_H_
#include <iostream>
#include <mutex>
#include <thread>
#include "linked_list.h"

// basic node structure
class linked_list {
  class node {
   public:
    node(int key) : key(key) { next = nullptr; }
    int key;
    node *next;
  };

  node *head;

 public:
  linked_list() {
    this->head = new node(0);
  }

  bool contains(int key) {
    // key to left of key
    node *curr = head->next;
    while (curr && curr->key < key) {
      curr = curr->next;
    }
    // Check key is key
    if (curr && curr->key == key) {
      return true;  // success
    } else {
      return false;  // failure
    }
  }

  bool insert(int key) {
    // move to left of key
    node *tmp = new node(key);
    node *prev = head;  // dummy node
    node *curr = head->next;
    while (curr && curr->key < key) {
      prev = curr;
      curr = curr->next;
    }
    // insert key
    if (curr != nullptr && curr->key == key) { // item already here
      return false;
    } else { // than curr->key > tmp->key
      tmp->next = curr;
      prev->next = tmp;
      return true;
    }
  }

  bool remove(int key) {
    // if empty do nothing
    node *prev = head;
    if (prev->next == nullptr) {
      return false;
    }
    // move left of key
    node *curr = prev->next;
    while (curr && curr->key < key) {
      prev = curr;
      curr = curr->next;
    }
    // remove the key
    if (curr != nullptr && curr->key == key) {
      prev->next = curr->next;
      delete curr;
      return true;
    } else { // key is not exist
      return false;
    }
  }
};
#endif
