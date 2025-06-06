(ns refnn.core
  (:gen-class)
  (:require
   [clojure.string :as str]
   [refnn.array :as a]
   [refnn.engine :as e]
   [oz.core :as oz]))

(def ^:const reflector-model
  (let [x         (a/uniform -1 1 [1000 2])
        rm        [[-1.0 0.0] [0.0 1.0]]
        y         (a/transpose (a/dot rm (a/transpose x)))
        eta       0.1
        epoch     100
        model-id  :reflector-model
        params    {:train/epoch    epoch
                   :train/eta      eta
                   :model/a0       x
                   :model/y        y
                   :model/id       model-id
                   :neuron/layers  [2 10 2]
                   :neuron/limiter (partial - 0.5)}]
    (e/train params)))

(defn runner!
  [model test-x test-y & {mark-opts :mark-opts plot-opts :plot-opts}]
  (let [[X Y] (e/run-model {:model model :X test-x :Y test-y})]
    (oz/view!
     [:main
      [:h3 "This is a reflector model."]
      [:span "given a set of points it'd provide the opposite points along it's vertical axis,"]
      [:vega-lite
       {:mark (merge {:type "point" :shape "circle" :filled true :size 10} mark-opts)
        :data {:values (mapcat
                        (fn [tx ty ox oy]
                          [{:x tx :y ty :type "Input"}
                           {:x ox :y oy :type "Output"}])
                        test-x test-y X Y)}
        :encoding {:x {:field "x" :type "quantitative"}
                   :y {:field "y" :type "quantitative"}
                   :color {:field "type" :type "nominal"}}}
       (merge {:width 700 :height 500} plot-opts)]])))

(defn read-from-file [path]
  (->> (slurp path)
       (str/split-lines)
       (keep
        (fn [point]
          (let [[x y] (str/split point #",")]
            (when (and x y)
              {:x (parse-double x) :y (parse-double y)}))))))

(defn run1 "U" [model]
  (let [test-x (vec (range 0 1 0.01))
        test-y (mapv (fn [x] (Math/pow x 2)) test-x)]
    (runner! model test-x test-y)))

(defn run2 "V" [model]
  (let [test-x (vec (range 0 1 0.01))
        test-y (mapv dec test-x)]
    (runner! model test-x test-y)))

(defn run3 "u" [model]
  (let [test-x (vec (range 0 1 0.01))
        test-y (mapv #(Math/pow % 4) test-x)]
    (runner! model test-x test-y)))

(defn run4 "n" [model]
  (let [test-x (vec (range 0 1 0.01))
        test-y (mapv #(Math/cos %) test-x)]
    (runner! model test-x test-y)))

(defn run-face! "FACE" [model filepath]
  (let [fcs (read-from-file filepath)
        test-x (mapv :x fcs)
        test-y (mapv :y fcs)]
    (runner! model test-x test-y 
             :mark-opts {:shape "cross" :size 20}
             :plot-opts {:width 800 :height 600})))

(comment
  (oz/start-server!)
  (run1 reflector-model)
  (run2 reflector-model)
  (run3 reflector-model)
  (run4 reflector-model)
  (run-face! reflector-model "resources/face.csv")
  :rcf)

(defn greet
  "Callable entry point to the application."
  [data]
  (println (str "Hello, " (or (:name data) "World") "!")))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (run2 reflector-model)
  (greet {:name (first args)}))
