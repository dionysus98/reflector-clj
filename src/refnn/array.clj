(ns refnn.array)

(defn random
  "Generates a rows x cols matrix of random floats in [0,1). Optionally applies a limiter function to each element."
  [rows cols & {:keys [limiter]}]
  (mapv
   (fn [_]
     (cond->> (into [] (repeatedly cols rand))
       (fn? limiter) (mapv limiter)))
   (range rows)))

(defn uniform
  "Generates a matrix or vector of random floats in [low, high). If size is nil, returns a single value. If size is a number, returns a vector. If size is a sequence, returns a nested structure."
  ([low high] (uniform low high nil))
  ([low high size]
   (cond
     (number? size)
     (mapv (fn [_] (uniform low high)) (range size))

     (empty? size)
     (+ low (* (- high low) (rand)))

     (seq size)
     (cond
       (= (count size) 1)
       (uniform low high (first size))

       :else
       (mapv (fn [_] (uniform low high (rest size))) (range (first size)))))))

(defn shape
  "Returns the shape of a matrix as [rows cols]. For 1D vectors, cols is 0."
  [mtx]
  [(count mtx)
   (or (when (sequential? (first mtx)) (count (first mtx))) 0)])

(defn is-1d?
  "Returns true if the matrix is 1D (either dimension is zero)."
  [mtx]
  (boolean (some zero? (shape mtx))))

(defn is-2d?
  "Returns true if the matrix is 2D (both dimensions > 0)."
  [mtx]
  (every? (partial < 0) (shape mtx)))

(defn emap
  "Elementwise map of xfn over a 2D matrix mx."
  [xfn mx]
  (mapv
   (partial mapv xfn)
   mx))

(defn efn
  "Elementwise map of xfn over two matrices mx1 and mx2. Returns nil if shapes do not match."
  [xfn mx1 mx2]
  (when (= (shape mx1) (shape mx2))
    (mapv
     (partial mapv xfn)
     mx1 mx2)))

(defn transpose
  "Transposes a 2D matrix."
  [mtx]
  (if (is-1d? mtx)
    mtx
    (let [base (-> mtx first count (repeat []))
          tfn  (fn transposer
                 [acc row]
                 (map-indexed
                  (fn [idx item]
                    (conj item (nth row idx)))
                  acc))]
      (into [] (reduce tfn base mtx)))))

(defn reshape
  "Reshapes a matrix to [rows cols]."
  ([mtx rows] (reshape mtx rows 1))
  ([mtx rows cols]
   (cond
     (= (shape mtx) [rows cols])
     mtx

     (and (is-1d? mtx) (= (count mtx) (* rows cols)))
     (vec (partition cols mtx))

     (and (is-2d? mtx) (= (apply * (shape mtx)) (* rows cols)))
     (vec (partition cols (flatten mtx))))))

(defn dot
  "Matrix multiplication of mtx1 and mtx2. Returns nil if shapes are incompatible."
  [mtx1 mtx2]
  (let [[_ c1] (shape mtx1)
        [r2 _] (shape mtx2)]
    (when (= c1 r2)
      (let [mtx2-T (transpose mtx2)
            mmul (fn [r c] (apply + (mapv * r c)))]
        (mapv
         (fn [row] (mapv (partial mmul row) mtx2-T))
         mtx1)))))

(defn zeros
  "Returns a rows x cols matrix of zeros."
  [rows cols]
  (let [r> (fn [_] (into [] (repeat cols 0)))]
    (mapv r> (range rows))))

(defn column-stack
  "Stacks vectors as columns."
  [& args]
  (apply mapv vector args))

