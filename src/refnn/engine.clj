(ns refnn.engine
  (:require [refnn.array :as a]))

(defn reLU [z]
  (a/emap
   (partial max 0)
   z))

(defn reLU-prime [z]
  (a/emap
   (fn [v] (if (> v 0) 1 0))
   z))

(defn linear [z]
  z)

(defn linear-prime [z]
  (a/emap (constantly 1) z))

(defn update-row [mtx xfn row]
  (let [[r] (a/shape mtx)]
    (when (= r (count row))
      (reduce
       (fn [acc [idx v]]
         (update acc idx (partial mapv (partial xfn v))))
       mtx
       (map-indexed (comp identity vector) row)))))

(defn initialzie
  [[in-count hidden-count out-count] {:keys [limiter] :as params}]
  (let [w1 (a/random hidden-count in-count :limiter limiter)
        b1 (a/zeros hidden-count 1)
        w2 (a/random out-count hidden-count :limiter limiter)
        b2 (a/zeros out-count 1)]
    (merge (dissoc params :limiter) {:w1 w1 :b1 b1 :w2 w2 :b2 b2})))

(defn forward-prop [{:keys [a0 w1 b1 w2 b2] :as params}]
  (let [z1 (update-row (a/dot w1 a0) + (first (a/transpose b1)))
        a1 (reLU z1)
        z2 (update-row (a/dot w2 a1) + (first (a/transpose b2)))
        a2 (linear z2)]
    (merge params {:z1 z1 :a1 a1 :z2 z2 :a2 a2})))s

(defn compute-gradient
  [{:keys [z1 a1 w2 z2 a2 a0 y] :as params}]
  (let [db2 (a/efn * (a/efn - a2 y) (linear-prime z2))
        dw2 (a/dot db2 (a/transpose a1))
        da1 (a/dot (a/transpose w2) db2)
        db1 (a/efn * da1 (reLU-prime z1))
        dw1 (a/dot db1 (a/transpose a0))]
    (merge params {:dw1 dw1 :db1 db1 :dw2 dw2 :db2 db2})))

(defn update-wb
  [{:keys [w1 dw1 b1 db1 w2 dw2 b2 db2 eta] :as params}]
  (let [wb> (fn [x dx] (a/efn - x (a/emap (partial * eta) dx)))
        >w1 (wb> w1 dw1)
        >b1 (wb> b1 db1)
        >w2 (wb> w2 dw2)
        >b2 (wb> b2 db2)]
    (merge params {:w1 >w1 :b1 >b1 :w2 >w2 :b2 >b2})))

(defn cost
  [{:keys [a2 y]}]
  (some->> (a/efn - a2 y)
           (a/emap #(Math/pow % 2))
           flatten
           seq
           (apply +)
           (* 0.5)))

(defn log-cost->return!
  [params]
  (println (cost params))
  params)

(defn- train-current-epoch
  [a0 y state]
  (loop [X a0
         Y y
         S state]
    (if (seq X)
      (recur
       (rest X)
       (rest Y)
       (-> S
           (assoc :a0 (-> X first a/transpose (a/reshape 2 1))
                  :y  (-> Y first a/transpose (a/reshape 2 1)))
           forward-prop
           compute-gradient
           update-wb
           (cond-> (:debug? state) log-cost->return!)))
      S)))

(defn train
  [{:train/keys  [eta epoch debug?]
    :model/keys  [a0 y id]
    :neuron/keys [layers limiter]}]
  (loop [epoch epoch
         state  (initialzie
                 layers {:eta     eta
                         :limiter limiter
                         :debug?  debug?})]
    (if (> epoch 0)
      (recur
       (dec epoch)
       (train-current-epoch a0 y state))
      (-> state
          (assoc :model/id id :model/X a0 :model/Y y)
          (dissoc :limiter)))))

(defn run-model
  [{:keys [model X Y]}]
  (loop [state (a/column-stack X Y)
         model model
         o-x   []
         o-y   []]
    (if (seq state)
      (let [a0   (-> state first a/transpose (a/reshape 2 1))
            fpst (-> model
                     (assoc :a0 a0)
                     forward-prop)
            out  (:a2 fpst)]
        (recur (rest state)
               fpst
               (conj o-x (get-in out [0 0]))
               (conj o-y (get-in out [1 0]))))
      [o-x o-y])))

(comment
  ;; manual test 1
  (let [in  (initialzie
             [2 3 2]
             {:eta     0.01
              :a0      [[0.5] [0.5]]
              :y       [[-0.5] [0.5]]
              :limiter (partial - 0.5)})
        fp  (forward-prop in)
        cg  (compute-gradient fp)
        uwb (update-wb cg)]
    (select-keys uwb [:w1 :w2]))

  ;; Train test 1
  (def trained-model
    (let [a0    (a/uniform -1 1 [1000 2])
          rm    [[-1.0 0.0] [0.0 1.0]]
          y     (a/transpose (a/dot rm (a/transpose a0)))
          eta   0.01
          epoch 100
          res   (train
                 {:train/epoch    epoch
                  :train/eta      eta
                  :model/a0       a0
                  :model/y        y
                  :neuron/layers  [2 10 2]
                  :neuron/limiter (partial - 0.5)})]
      res))

  (time (let [a0    (a/uniform -1 1 [1000 2])
              rm    [[-1.0 0.0] [0.0 1.0]]
              y     (a/transpose (a/dot rm (a/transpose a0)))
              eta   0.01
              epoch 100
              res   (train
                     {:train/epoch    epoch
                      :train/eta      eta
                      :model/a0       a0
                      :model/y        y
                      :neuron/layers  [2 10 2]
                      :neuron/limiter (partial - 0.5)})]
          (:a2 res)))

  :rcf)
