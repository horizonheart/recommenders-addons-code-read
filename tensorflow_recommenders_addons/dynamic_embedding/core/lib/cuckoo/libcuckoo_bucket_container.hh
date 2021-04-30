#ifndef LIBCUCKOO_BUCKET_CONTAINER_H
#define LIBCUCKOO_BUCKET_CONTAINER_H

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

#include "cuckoohash_util.hh"

/**
 * libcuckoo_bucket_container manages storage of key-value pairs for the table.
 * It stores the items inline in uninitialized memory, and keeps track of which
 * slots have live data and which do not. It also stores a partial hash for
 * each live key. It is sized by powers of two.
 *
 * @tparam Key type of keys in the table
 * @tparam T type of values in the table
 * @tparam Allocator type of key-value pair allocator
 * @tparam Partial type of partial keys
 * @tparam SLOT_PER_BUCKET number of slots for each bucket in the table
 */
// T ： typedef gtl::InlinedVector<V, 4> ValueArray
// todo Allocator = std::allocator<std::pair<const Key, T>>
// todo 存放hash值的container容器
template <class Key, class T, class Allocator, class Partial,
          std::size_t SLOT_PER_BUCKET>
class libcuckoo_bucket_container {
 public:
  using key_type = Key;
  using mapped_type = T;
  using value_type = std::pair<const Key, T>;

 private:
//  rebind用来从已有的allocator类型获取一个新的用来分配另一个类型U的allocator类型
//  traits_ = allocator_traits<std::allocator<std::pair<const Key, T>>>
  using traits_ = typename std::allocator_traits<
      Allocator>::template rebind_traits<value_type>;

 public:
  using allocator_type = typename traits_::allocator_type; // std::allocator<std::pair<const Key, T>>
  using partial_t = Partial;
  using size_type = typename traits_::size_type;
  using reference = value_type &;
  using const_reference = const value_type &;
  using pointer = typename traits_::pointer; // std::pair<const Key, T> *
  using const_pointer = typename traits_::const_pointer;

  /*
   * The bucket type holds SLOT_PER_BUCKET key-value pairs, along with their
   * partial keys and occupancy info. It uses aligned_storage arrays to store
   * the keys and values to allow constructing and destroying key-value pairs
   * in place. The lifetime of bucket data should be managed by the container.
   * It is the user's responsibility to confirm whether the data they are
   * accessing is live or not.
   */
  //todo 存储桶的真实数据
  class bucket {
   public:
    // 空的构造函数
    bucket() noexcept : occupied_{} {}

    // 取出常量值
    const value_type &kvpair(size_type ind) const {
      return *static_cast<const value_type *>(
          static_cast<const void *>(&values_[ind]));
    }

    //取值
    value_type &kvpair(size_type ind) {
      return *static_cast<value_type *>(static_cast<void *>(&values_[ind]));
    }

    // 根据索引取出对应的key
    const key_type &key(size_type ind) const {
      return storage_kvpair(ind).first;
    }


    key_type &&movable_key(size_type ind) {
      return std::move(storage_kvpair(ind).first);
    }

    // 取出对应的value
    const mapped_type &mapped(size_type ind) const {
      return storage_kvpair(ind).second;
    }

    mapped_type &mapped(size_type ind) { return storage_kvpair(ind).second; }

    partial_t partial(size_type ind) const { return partials_[ind]; }
    partial_t &partial(size_type ind) { return partials_[ind]; }

    bool occupied(size_type ind) const { return occupied_[ind]; }
    bool &occupied(size_type ind) { return occupied_[ind]; }

   private:
    // 友元类
    friend class libcuckoo_bucket_container;

    // todo 定义存储值的类型
    using storage_value_type = std::pair<Key, T>;

    // todo 隐藏接口
    const storage_value_type &storage_kvpair(size_type ind) const {
      return *static_cast<const storage_value_type *>(
          static_cast<const void *>(&values_[ind]));
    }

    storage_value_type &storage_kvpair(size_type ind) {
      return *static_cast<storage_value_type *>(
          static_cast<void *>(&values_[ind]));
    }
    // todo 真正存储值的集合
    // std::aligned_storage可以看成一个内存对其的缓冲区
    // std::array大小不可变，第二个参数是数组的大小
    std::array<typename std::aligned_storage<sizeof(storage_value_type),
                                             alignof(storage_value_type)>::type,
               SLOT_PER_BUCKET>
        values_;
    std::array<partial_t, SLOT_PER_BUCKET> partials_;
    std::array<bool, SLOT_PER_BUCKET> occupied_;
  };

  // todo 构造函数，创建内存空间
  // allocator_type = std::allocator<std::pair<const Key, T>>
  libcuckoo_bucket_container(size_type hp, const allocator_type &allocator)
      : allocator_(allocator), // allocator =  std::allocator<std::pair<const Key, T>>
        bucket_allocator_(allocator), //bucket_allocator_ = std::allocator<bucket> 
        hashpower_(hp),
        //buckets_ 为bucket * 指针类型，只分配指定的大小
        buckets_(bucket_allocator_.allocate(size())) {
    // The bucket default constructor is nothrow, so we don't have to
    // worry about dealing with exceptions when constructing all the
    // elements.
    static_assert(std::is_nothrow_constructible<bucket>::value,
                  "libcuckoo_bucket_container requires bucket to be nothrow "
                  "constructible");
    // 逐个桶创建对象
    for (size_type i = 0; i < size(); ++i) {
      //  traits_ = allocator_traits<std::allocator<std::pair<const Key, T>>>
      traits_::construct(allocator_, &buckets_[i]);
    }
  }

  // 析构函数
  ~libcuckoo_bucket_container() noexcept { destroy_buckets(); }

  // todo 拷贝构造函数
  libcuckoo_bucket_container(const libcuckoo_bucket_container &bc)
      : allocator_(
            traits_::select_on_container_copy_construction(bc.allocator_)),
        bucket_allocator_(allocator_),
        hashpower_(bc.hashpower()),
        buckets_(transfer(bc.hashpower(), bc, std::false_type())) {}

  libcuckoo_bucket_container(const libcuckoo_bucket_container &bc,
                             const allocator_type &a)
      : allocator_(a),
        bucket_allocator_(allocator_),
        hashpower_(bc.hashpower()),
        buckets_(transfer(bc.hashpower(), bc, std::false_type())) {}

  // todo 移动拷贝构造函数
  libcuckoo_bucket_container(libcuckoo_bucket_container &&bc)
      : allocator_(std::move(bc.allocator_)),
        bucket_allocator_(allocator_),
        hashpower_(bc.hashpower()),
        buckets_(std::move(bc.buckets_)) {
    // De-activate the other buckets container
    bc.buckets_ = nullptr;
  }

  libcuckoo_bucket_container(libcuckoo_bucket_container &&bc,
                             const allocator_type &a)
      : allocator_(a), bucket_allocator_(allocator_) {
    move_assign(bc, std::false_type());
  }

  // todo 重载预算法
  libcuckoo_bucket_container &operator=(const libcuckoo_bucket_container &bc) {
    destroy_buckets();
    copy_allocator(allocator_, bc.allocator_,
                   typename traits_::propagate_on_container_copy_assignment());
    bucket_allocator_ = allocator_;
    hashpower(bc.hashpower());
    buckets_ = transfer(bc.hashpower(), bc, std::false_type());
    return *this;
  }

  libcuckoo_bucket_container &operator=(libcuckoo_bucket_container &&bc) {
    destroy_buckets();
    move_assign(bc, typename traits_::propagate_on_container_move_assignment());
    return *this;
  }

  // todo 交换值，这里的bc相当于src
  void swap(libcuckoo_bucket_container &bc) noexcept {
    swap_allocator(allocator_, bc.allocator_,
                   typename traits_::propagate_on_container_swap());
    swap_allocator(bucket_allocator_, bc.bucket_allocator_,
                   typename traits_::propagate_on_container_swap());
    // Regardless of whether we actually swapped the allocators or not, it will
    // always be okay to do the remainder of the swap. This is because if the
    // allocators were swapped, then the subsequent operations are okay. If the
    // allocators weren't swapped but compare equal, then we're okay. If they
    // weren't swapped and compare unequal, then behavior is undefined, so
    // we're okay.
    size_t bc_hashpower = bc.hashpower();
    bc.hashpower(hashpower());
    hashpower(bc_hashpower);
    std::swap(buckets_, bc.buckets_);
  }

  // todo 返回hashpower的大小
  size_type hashpower() const {
    return hashpower_.load(std::memory_order_acquire);
  }

  // todo 设置hashpower的大小
  void hashpower(size_type val) {
    hashpower_.store(val, std::memory_order_release);
  }

  // todo 计算hash表的大小
  size_type size() const { return size_type(1) << hashpower(); }

  // todo 返回allocator的类型
  allocator_type get_allocator() const { return allocator_; }

  // todo 中括号运算符重载
  bucket &operator[](size_type i) { return buckets_[i]; }
  
  const bucket &operator[](size_type i) const { return buckets_[i]; }

  // todo 设置key value的值
  // Constructs live data in a bucket
  template <typename K, typename... Args>
  void setKV(size_type ind, size_type slot, partial_t p, K &&k,
             Args &&... args) {
    // 获取要插入的桶的索引
    bucket &b = buckets_[ind];
    assert(!b.occupied(slot));
    b.partial(slot) = p;
    // 为指定的内存赋值
    traits_::construct(allocator_, std::addressof(b.storage_kvpair(slot)),
                       std::piecewise_construct,
                       std::forward_as_tuple(std::forward<K>(k)),
                       std::forward_as_tuple(std::forward<Args>(args)...));
    // This must occur last, to enforce a strong exception guarantee
    b.occupied(slot) = true;
  }

  // Destroys live data in a bucket
  // todo 消除具体某个桶内的数据
  void eraseKV(size_type ind, size_type slot) {
    bucket &b = buckets_[ind];
    assert(b.occupied(slot));
    b.occupied(slot) = false;
    traits_::destroy(allocator_, std::addressof(b.storage_kvpair(slot)));
  }

  // Destroys all the live data in the buckets. Does not deallocate the bucket
  // memory.
  // todo 清除桶内所有活着的数据
  void clear() noexcept {
    static_assert(
        std::is_nothrow_destructible<key_type>::value &&
            std::is_nothrow_destructible<mapped_type>::value,
        "libcuckoo_bucket_container requires key and value to be nothrow "
        "destructible");
    for (size_type i = 0; i < size(); ++i) {
      bucket &b = buckets_[i];
      for (size_type j = 0; j < SLOT_PER_BUCKET; ++j) {
        if (b.occupied(j)) {
          eraseKV(i, j);
        }
      }
    }
  }

  // Destroys and deallocates all data in the buckets. After this operation,
  // the bucket container will have no allocated data. It is still valid to
  // swap, move or copy assign to this container.
  void clear_and_deallocate() noexcept { destroy_buckets(); }

 private:
 //  todo 属性值 traits_ = allocator_traits<std::allocator<std::pair<const Key, T>>>
  using bucket_traits_ = typename traits_::template rebind_traits<bucket>;
  // bucket_pointer 为 bucket *
  using bucket_pointer = typename bucket_traits_::pointer;

  // todo 拷贝函数
  // true here means the allocators from `src` are propagated on libcuckoo_copy
  template <typename A>
  void copy_allocator(A &dst, const A &src, std::true_type) {
    dst = src;
  }

  template <typename A>
  void copy_allocator(A &dst, const A &src, std::false_type) {}

  // TODO 交换函数
  // true here means the allocators from `src` are propagated on libcuckoo_swap
  template <typename A>
  void swap_allocator(A &dst, A &src, std::true_type) {
    std::swap(dst, src);
  }

  template <typename A>
  void swap_allocator(A &, A &, std::false_type) {}

  // true here means the bucket allocator should be propagated
  // todo 移动指针
  void move_assign(libcuckoo_bucket_container &src, std::true_type) {
    allocator_ = std::move(src.allocator_);
    bucket_allocator_ = allocator_;
    hashpower(src.hashpower());
    buckets_ = src.buckets_;
    src.buckets_ = nullptr;
  }

  void move_assign(libcuckoo_bucket_container &src, std::false_type) {
    hashpower(src.hashpower());
    if (allocator_ == src.allocator_) {
      buckets_ = src.buckets_;
      src.buckets_ = nullptr;
    } else {
      buckets_ = transfer(src.hashpower(), src, std::true_type());
    }
  }

  // 销毁内存桶
  void destroy_buckets() noexcept {
    if (buckets_ == nullptr) {
      return;
    }
    // The bucket default constructor is nothrow, so we don't have to
    // worry about dealing with exceptions when constructing all the
    // elements.
    static_assert(std::is_nothrow_destructible<bucket>::value,
                  "libcuckoo_bucket_container requires bucket to be nothrow "
                  "destructible");
    // 先删除桶内集合所占用的内存
    clear();
    // 再删除桶占的内存，调用桶的析构函数
    for (size_type i = 0; i < size(); ++i) {
      traits_::destroy(allocator_, &buckets_[i]);
    }

    // 释放桶对应的空间
    bucket_allocator_.deallocate(buckets_, size());
    buckets_ = nullptr;
  }

  // todo 移动或拷贝值
  // `true` here refers to whether or not we should move
  void move_or_copy(size_type dst_ind, size_type dst_slot, bucket &src,
                    size_type src_slot, std::true_type) {
    setKV(dst_ind, dst_slot, src.partial(src_slot), src.movable_key(src_slot),
          std::move(src.mapped(src_slot)));
  }

  // todo 移动或拷贝值
  void move_or_copy(size_type dst_ind, size_type dst_slot, bucket &src,
                    size_type src_slot, std::false_type) {
    setKV(dst_ind, dst_slot, src.partial(src_slot), src.key(src_slot),
          src.mapped(src_slot));
  }

  // todo  迁移
  template <bool B>
  bucket_pointer transfer(
      size_type dst_hp,
      typename std::conditional<B, libcuckoo_bucket_container &,
                                const libcuckoo_bucket_container &>::type src,
      std::integral_constant<bool, B> move) {
    assert(dst_hp >= src.hashpower());
    libcuckoo_bucket_container dst(dst_hp, get_allocator());
    // Move/copy all occupied slots of the source buckets
    for (size_t i = 0; i < src.size(); ++i) {
      for (size_t j = 0; j < SLOT_PER_BUCKET; ++j) {
        if (src.buckets_[i].occupied(j)) {
          dst.move_or_copy(i, j, src.buckets_[i], j, move);
        }
      }
    }
    // Take away the pointer from `dst` and return it
    bucket_pointer dst_pointer = dst.buckets_;
    dst.buckets_ = nullptr;
    return dst_pointer;
  }

  // This allocator matches the value_type, but is not used to construct
  // storage_value_type pairs, or allocate buckets
  //todo allocator_type=std::allocator<std::pair<const Key, T>>
  allocator_type allocator_;
  // This allocator is used for actually allocating buckets. It is simply
  // copy-constructed from `allocator_`, and will always be copied whenever
  // allocator_ is copied.

  // todo traits_ = allocator_traits<std::allocator<std::pair<const Key, T>>>
  // 重新绑定分配的类型
  // std::allocator<bucket>
  typename traits_::template rebind_alloc<bucket> bucket_allocator_;
  // This needs to be atomic, since it can be read and written by multiple
  // threads not necessarily synchronized by a lock.
  std::atomic<size_type> hashpower_;
  // These buckets are protected by striped locks (external to the
  // BucketContainer), which must be obtained before accessing a bucket.
  bucket_pointer buckets_;

  // If the key and value are Trivial, the bucket be serilizable. Since we
  // already disallow user-specialized instances of std::pair, we know that the
  // default implementation of std::pair uses a default copy constructor, so
  // this should be okay. We could in theory just check if the type is
  // TriviallyCopyable but this check is not available on some compilers we
  // want to support.
  
  // TODO  运算符重载
  template <typename ThisKey, typename ThisT>
  friend typename std::enable_if<std::is_trivial<ThisKey>::value &&
                                     std::is_trivial<ThisT>::value,
                                 std::ostream &>::type
  operator<<(std::ostream &os,
             const libcuckoo_bucket_container<ThisKey, ThisT, Allocator,
                                              Partial, SLOT_PER_BUCKET> &bc) {
    size_type hp = bc.hashpower();
    os.write(reinterpret_cast<const char *>(&hp), sizeof(size_type));
    os.write(reinterpret_cast<const char *>(bc.buckets_),
             sizeof(bucket) * bc.size());
    return os;
  }

  // todo 重载读取的运算符
  template <typename ThisKey, typename ThisT>
  friend typename std::enable_if<std::is_trivial<ThisKey>::value &&
                                     std::is_trivial<ThisT>::value,
                                 std::istream &>::type
  operator>>(std::istream &is,
             libcuckoo_bucket_container<ThisKey, ThisT, Allocator, Partial,
                                        SLOT_PER_BUCKET> &bc) {
    size_type hp;
    is.read(reinterpret_cast<char *>(&hp), sizeof(size_type));
    // 创建一个临时的桶
    libcuckoo_bucket_container new_bc(hp, bc.get_allocator());
    // 将流里面的内容读取到bc桶里面
    is.read(reinterpret_cast<char *>(new_bc.buckets_),
            new_bc.size() * sizeof(bucket));
    bc.swap(new_bc);
    return is;
  }
};
#endif  // LIBCUCKOO_BUCKET_CONTAINER_H
