#pragma once

#include "common.h"

namespace xforce {

template <
  typename Key, 
  typename Val, 
  typename KeyEqualF = std::equal_to<Key>,
  typename HashF = std::tr1::hash<Key> >
class CloseHashmap {
 private:
  typedef CloseHashmap<Key, Val, KeyEqualF, HashF> Self;
  typedef std::pair<int8_t, std::pair<Key, Val> > Item;
  
 public: 
  class Iterator {
    public:
     typedef CloseHashmap<Key, Val, KeyEqualF, HashF> Master;
     typedef Iterator Self;

    public:
     Iterator();
     Iterator(const Master& master, bool begin);
     Iterator(const Iterator& iter);

     inline void Next();
     bool IsEnd() const { return pos_ == master_->Capacity(); }
     inline std::pair<Key, Val>* operator->();
     Self& operator=(const Iterator& iter);
     inline bool operator==(const Iterator& iter) const;
     bool operator!=(const Iterator& iter) const { return !operator==(iter); }

    private:
     const Master* master_;
     size_t pos_;
  };

 private:
  static const size_t kNumPrimes=28; 
  static const size_t kPrimes[kNumPrimes];
  static const double kResizeFactor;

 public:
  bool Init(
      size_t size_array, 
      bool to_resize = true,
      KeyEqualF key_equal = KeyEqualF(), 
      HashF hash = HashF());

  inline bool Insert(const Key& key, const Val& val);
  inline bool Upsert(const Key& key, const Val& val);
  inline bool Erase(const Key& key);
  inline void Copy(const Self &from);

  inline const Val* Get(const Key& key) const;
  inline Val* Get(const Key& key);
  size_t Size() const { return num_elm_; }
  size_t Capacity() const { return size_array_; }

  Iterator Begin() { return Iterator(*this, true); }
  Iterator End() { return Iterator(*this, false); }

  virtual ~CloseHashmap();
 
 private:
  inline const Item* GetItemOrNull_(const Key& key) const;
  inline Item* GetItemOrNull_(const Key& key);
  inline const Item* GetItemOrFree_(const Key& key) const;
  inline Item* GetItemOrFree_(const Key& key);
  inline void Resize();

 private:
  size_t size_array_;
  bool to_resize_;
  KeyEqualF key_equal_;
  HashF hash_;

  Item *items_;
  size_t num_elm_;
};

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
const double CloseHashmap<Key, Val, KeyEqualF, HashF>::kResizeFactor = 1.5;

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
CloseHashmap<Key, Val, KeyEqualF, HashF>::Iterator::Iterator() :
  master_(NULL),
  pos_(0) {}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
CloseHashmap<Key, Val, KeyEqualF, HashF>::Iterator::Iterator(
    const Master& master,
    bool begin) :
  master_(&master) {
  if (!begin) {
    pos_ = master_->Capacity();
  } else {
    for (pos_=0; pos_ < master_->Capacity(); ++pos_) {
      if (master_->items_[pos_].first>0) {
        return;
      }
    }
  }
}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
CloseHashmap<Key, Val, KeyEqualF, HashF>::Iterator::Iterator(
    const Iterator& iter) :
  master_(iter.master_),
  pos_(iter.pos_) {}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
void CloseHashmap<Key, Val, KeyEqualF, HashF>::Iterator::Next() {
  if (master_->Capacity() == pos_) {
    return;
  }

  for (++pos_; pos_ < master_->Capacity(); ++pos_) {
    if (master_->items_[pos_].first>0) {
      return;
    }
  }
}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
std::pair<Key, Val>* CloseHashmap<Key, Val, KeyEqualF, HashF>::Iterator::operator->() {
  return pos_ != master_->Capacity() ? 
    &(master_->items_[pos_].second) :
    NULL;
}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
typename CloseHashmap<Key, Val, KeyEqualF, HashF>::Iterator& 
CloseHashmap<Key, Val, KeyEqualF, HashF>::Iterator::operator=(const Iterator& iter) {
  master_ = iter.master_;
  pos_ = iter.pos_;
  return *this;
}
template <typename Key, typename Val, typename KeyEqualF, typename HashF>
bool CloseHashmap<Key, Val, KeyEqualF, HashF>::Iterator::operator==(const Iterator& iter) const {
  return master_ == iter.master_ && pos_ == iter.pos_;
}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
const size_t CloseHashmap<Key, Val, KeyEqualF, HashF>::kPrimes[kNumPrimes] = {
    53,         97,         193,       389,       769,
    1543,       3079,       6151,      12289,     24593,
    49157,      98317,      196613,    393241,    786433,
    1572869,    3145739,    6291469,   12582917,  25165843,
    50331653,   100663319,  201326611, 402653189, 805306457,
    1610612741, 3221225473, 4294967291,
};

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
bool CloseHashmap<Key, Val, KeyEqualF, HashF>::Init(
    size_t size_array,
    bool to_resize,
    KeyEqualF key_equal,
    HashF hash) {
  for (size_t i=0; i<kNumPrimes; ++i) {
    if ( 0 == size_array % kPrimes[i] ) {
      return false;
    }
  }

  size_array_=size_array;
  to_resize_=to_resize;
  key_equal_=key_equal;
  hash_=hash;
  num_elm_=0;
  XFC_MALLOC(items_, Item*, sizeof(Item) * size_array_);
  bzero(items_, size_array_ * sizeof(Item));
  return true;
}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
bool CloseHashmap<Key, Val, KeyEqualF, HashF>::Insert(const Key& key, const Val& val) {
  if (unlikely(size_array_==num_elm_)) {
    if (to_resize_) {
      Resize();
    } else {
      return false;
    }
  }

  Item* item = GetItemOrFree_(key);
  if (item->first <= 0) {
    *item = std::make_pair(1, std::make_pair(key, val));
    ++num_elm_;
    return true;
  } else {
    return false;
  }
}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
bool CloseHashmap<Key, Val, KeyEqualF, HashF>::Upsert(const Key& key, const Val& val) {
  Item *item = GetItemOrFree_(key);
  if (item != NULL && item->first > 0) {
    *item = std::make_pair(1, std::make_pair(key, val));
    return true;
  }

  if (size_array_!=num_elm_) {
    *item = std::make_pair(1, std::make_pair(key, val));
    ++num_elm_;
    return true;
  } else {
    if (to_resize_) {
      Resize();
      Item* item = GetItemOrFree_(key);
      *item = std::make_pair(1, std::make_pair(key, val));
      ++num_elm_;
      return true;
    } else {
      return false;
    }
  }  
}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
bool CloseHashmap<Key, Val, KeyEqualF, HashF>::Erase(const Key& key) {
  Item* item = GetItemOrNull_(key);
  if (NULL!=item) {
    item->first = -1;
    --num_elm_;
    return true;
  } else {
    return false;
  }
}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
void CloseHashmap<Key, Val, KeyEqualF, HashF>::Copy(const Self &other) {
  if (size_array_ != other.size_array_) {
    XFC_REALLOC(items_, items_, Item*, other.size_array_ * sizeof(Item));
    size_array_ = other.size_array_;
  }
  num_elm_ = other.num_elm_;
  memcpy(items_, other.items_, num_elm_ * sizeof(Item));
}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
const Val* CloseHashmap<Key, Val, KeyEqualF, HashF>::Get(const Key& key) const {
  const Item* item = GetItemOrNull_(key);
  return NULL!=item ? &(item->second.second) : NULL;
}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
Val* CloseHashmap<Key, Val, KeyEqualF, HashF>::Get(const Key& key) {
  return CCAST<Val*>(CCAST<const Self*>(this)->Get(key));
}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
const typename CloseHashmap<Key, Val, KeyEqualF, HashF>::Item* 
CloseHashmap<Key, Val, KeyEqualF, HashF>::GetItemOrNull_(const Key& key) const {
  size_t pos = hash_(key) % size_array_;
  if (items_[pos].first > 0 && key_equal_(key, items_[pos].second.first)) {
    return &(items_[pos]);
  } else if (items_[pos].first == 0) {
    return NULL;
  }

  const size_t step = kPrimes[pos%kNumPrimes]; 
  for (size_t i=0; i<size_array_-1; ++i) {
    pos = (pos+step) % size_array_;
    if (items_[pos].first > 0 && key_equal_(key, items_[pos].second.first)) {
      return &(items_[pos]);
    } else if (items_[pos].first == 0) {
      return NULL;
    }
  }
  return NULL;
}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
const typename CloseHashmap<Key, Val, KeyEqualF, HashF>::Item* 
CloseHashmap<Key, Val, KeyEqualF, HashF>::GetItemOrFree_(const Key& key) const {
  const Item* item = GetItemOrNull_(key);
  if (item!=NULL) {
    return item;
  }

  size_t pos = hash_(key) % size_array_;
  const size_t step = kPrimes[pos%kNumPrimes]; 
  for (size_t i=0; i<size_array_; ++i) {
    if (items_[pos].first<=0) {
      return &(items_[pos]);
    }
    pos = (pos+step) % size_array_;
  }
  return NULL;
}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
typename CloseHashmap<Key, Val, KeyEqualF, HashF>::Item* 
CloseHashmap<Key, Val, KeyEqualF, HashF>::GetItemOrNull_(const Key& key) {
  return CCAST<Item*>(CCAST<const Self*>(this)->GetItemOrNull_(key));
}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
typename CloseHashmap<Key, Val, KeyEqualF, HashF>::Item* 
CloseHashmap<Key, Val, KeyEqualF, HashF>::GetItemOrFree_(const Key& key) {
  return CCAST<Item*>(CCAST<const Self*>(this)->GetItemOrFree_(key));
}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
void CloseHashmap<Key, Val, KeyEqualF, HashF>::Resize() {
  size_t new_size_array = kResizeFactor * size_array_;  
  if (new_size_array == size_array_) {
    ++new_size_array;
  }

  Item *new_items;
  XFC_MALLOC(new_items, Item*, sizeof(Item) * new_size_array);
  bzero(new_items, new_size_array * sizeof(Item));

  for (size_t i=0; i<size_array_; ++i) {
    if (items_[i].first > 0) {
      const Key &key = items_[i].second.first;
      size_t new_pos = hash_(key) % new_size_array;  
      if (new_items[new_pos].first == 0) {
        new_items[new_pos] = items_[i];
        continue;
      }

      const size_t step = kPrimes[new_pos%kNumPrimes]; 
      for (size_t j=0; j < new_size_array-1; ++j) {
        new_pos = (new_pos+step) % new_size_array;
        if (new_items[new_pos].first == 0) {
          new_items[new_pos] = items_[i];
          break;
        }
      }
    }
  }

  size_array_ = new_size_array;
  XFC_FREE(items_)
  items_ = new_items;
}

template <typename Key, typename Val, typename KeyEqualF, typename HashF>
CloseHashmap<Key, Val, KeyEqualF, HashF>::~CloseHashmap() {
  XFC_FREE(items_)
}

}
