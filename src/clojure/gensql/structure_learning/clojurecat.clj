(ns gensql.structure-learning.clojurecat
  "Utility functions related to the ClojureCat command-line interface."
  (:refer-clojure :exclude [merge])
  (:require [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [gensql.inference.gpm :as gpm]
            [gensql.inference.gpm.ensemble :as ensemble]))

(defn config
  "Outputs a config file for use with the clojurecat command-line interface."
  [{:keys [iterations]}]
  (pprint {:model :xcat
           :n-infer-iters iterations}))

(defn merge
  "Merges the GPMs in a directory."
  [{path :models out :out}]
  (let [path (str path)
        out (str out)
        data (->> (io/file path)
                  (file-seq)
                  (filter #(.isFile %))
                  (map slurp)
                  (map gpm/read-string)
                  ;; Its okay to call this constructor as we know there's at least 1 model
                  ;;   _and_ its already a sequence.
                  (ensemble/->Ensemble))]
    (with-open [writer (clojure.java.io/writer out)]
      (binding [*out* writer]
        (pr data)))))
