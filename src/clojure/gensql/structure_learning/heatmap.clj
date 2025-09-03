(ns gensql.structure-learning.heatmap
  "Functions for generating heatmaps."
  (:require [clojure.data.json :as json]
            [medley.core :as medley])
  (:import [smile.clustering HierarchicalClustering]
           [smile.clustering.linkage UPGMCLinkage]))

;; The purpose of this file is to generate dependence probability heatmaps from
;; a data structure called a statistic map. Statistic maps define the dependence
;; probability between all pairs of columns in a dataset. A statistic map is a
;; map of maps where the keys of both the outer map and the inner maps are
;; columns, and the values of the inner maps are the probability that the
;; columns are dependent or a map of statistics relating the two columns.

(defn- columns
  "Returns all the columns in a statistic map."
  [sm]
  (into (set (keys sm))
        (mapcat keys (vals sm))))

;; The challenge when generating a dependence probability heatmap is finding a
;; column ordering that will position columns that have a high probability of
;; being dependent next to one another. To generate this ordering we use
;; agglomerative clustering. The agglomerative clustering algorithm works as
;; follows:

;; We generate a tree of clusters from the bottom up. At the bottom of the tree
;; each item is in its own cluster. To generate the next level in the tree we
;; merge the two closest pairs of clusters, where closeness is determinted by a
;; distance metric we define. We repeat this process until we generate the top
;; of the tree, where there is only one cluster.

(defn- clustering-comp
  "Returns a comparator function for use with `clojure.core/sort` that will sort
  columns according to a statistic map."
  [sm]
  ;; We use an agglomerative clustering algorithm from the Smile machine
  ;; learning library. For our distance metric (the distance between columns) we
  ;; use the probability that the columns are dependent. Columns that have a
  ;; higer probability of dependence are considered closer together.
  (let [;; Smile's distance metrics require that you provide the distance
        ;; between each pair of values as a two-dimensional array of doubles,
        ;; where the value at coordinate (x, y) in the array is the distance
        ;; between the items at index x and index y.
        columns (vec (columns sm))
        linkage (UPGMCLinkage/of
                 (into-array
                  (for [c1 (range (count columns))]
                    (double-array
                     (for [c2 (range (count columns))]
                       (let [col1 (nth columns c1)
                             col2 (nth columns c2)
                             stat (get-in sm [col1 col2] ::not-found)]
                         (if (= ::not-found stat)
                           (throw (ex-info "No statistic for columns." {:columns [col1 col2]}))
                           stat)))))))
        clustering (HierarchicalClustering/fit linkage)]
    ;; The `HierarchicalClustering` class provides a method, `partition`, which
    ;; cuts the tree into n groups. `partition` returns an integer array where
    ;; each integer represents a unique cluster assignment for the item (in our
    ;; case, column) at that index. We define a comparator which cuts the tree
    ;; into successively larger sets of clusters until the columns of interest
    ;; are in different clusters. The comparator then compares the unique
    ;; integers for the two clusters.
    (fn [col1 col2]
      (loop [n-clusters 2]
        (if (>= n-clusters (count columns))
          false
          (let [clusters (vec (.partition clustering n-clusters))
                c1i (.indexOf columns col1)
                c2i (.indexOf columns col2)
                c1s (nth clusters c1i)
                c2s (nth clusters c2i)]
            (if (= c1s c2s)
              (recur (inc n-clusters))
              (> c1s c2s))))))))

(defn- sort-spec
  "Returns a Vega-Lite snippet that will cause columns in the x and y axes to be
  sorted according to the provided comparator."
  [sm]
  (let [columns (columns sm)
        clustering-comp (clustering-comp sm)
        order (->> columns
                   (sort clustering-comp)
                   (vec))]
    {:encoding {:x {:sort order}
                :y {:sort order}}}))

(defn- domain-spec
  "Returns a Vega-Lite snippet that sets the domain for the fill encoding
  channel."
  [domain]
  {:encoding {:fill {:scale {:domain domain}}}})

(defn- scheme-spec
  "Returns a Vega-Lite snippet that sets the color scheme for the fill encoding
  channel."
  [scheme]
  {:encoding {:fill {:scale {:scheme scheme}}}})

(defn- heatmap-spec
  "Returns a Vega-Lite snippet that renders the data as a heatmap, using fields
  \"col1\" and \"col2\" as the x and y axis, respectively. Accepts the
  quantitative field to use in the fill encoding as an argument."
  [field]
  {:mark {:type "rect"}
   :encoding {:x {:title nil
                  :field "col1"
                  :type "nominal"
                  :axis {:orient "top"
                         :labelAngle 315}}
              :y {:title nil
                  :field "col2"
                  :type "nominal"}
              :fill {:title nil
                     :field field
                     :type "quantitative"}}})

(defn- update-stats
  "Given a map of maps, calls f on each value of each inner map and args."
  [sm f & args]
  (update-vals sm (fn [inner] (update-vals inner #(apply f % args)))))

(defn- fill-missing
  "Given a map of maps, ensures that the outer maps and the inner maps have the
  same keys. Uses x as the value for any keys added to the inner maps."
  [sm x]
  (let [columns (columns sm)]
    (reduce (fn [sm [col1 col2]]
              (update-in sm [col1 col2] #(or % x)))
            sm
            (for [col1 columns
                  col2 columns]
              [col1 col2]))))

(defn- values
  "Transforms the provided statistic maps into a table, expressed as a vector of
  maps. Each map in the vector has keys for the two columns and one key for each
  of the statistics."
  [key->sm]
  (let [columns (into #{} (mapcat keys (vals key->sm)))]
    (for [col1 columns
          col2 columns]
      (reduce-kv (fn [row k sm]
                   (assoc row k (get-in sm [col1 col2])))
                 {:col1 col1
                  :col2 col2}
                 key->sm))))

(def ^:private vega-lite-schema "https://vega.github.io/schema/vega-lite/v5.json")

(defn- preserve-nan-as-keyword [json-str]
  (-> json-str
      (clojure.string/replace #"\bNaN\b" "\"__NaN__\"")
      (clojure.string/replace #"\bInfinity\b" "\"__Infinity__\"")
      (clojure.string/replace #"-Infinity\b" "\"__NegInfinity__\"")))

(defn- handle-nan [k v]
  (case v
    "__NaN__" Double/NaN
    "__Infinity__" Double/POSITIVE_INFINITY
    "__NegInfinity__" Double/NEGATIVE_INFINITY
    v))

(defn vega-lite
  "Prints a Vega-Lite spec for a statistic JSON file."
  [& {:keys [default domain field name scheme stats-path sort-path]}]
  (assert (some? name))
  (assert (some? stats-path))
  (let [sm (cond-> (preserve-nan-as-keyword (slurp stats-path))
             true (json/read-str :value-fn handle-nan)
             field (update-stats #(get % field))
             default (fill-missing default))
        sort-sm (some-> sort-path
                        (slurp)
                        (preserve-nan-as-keyword)
                        (json/read-str :value-fn handle-nan)
                        (fill-missing 1.))
        base-spec {:$schema vega-lite-schema
                   :data {:values (values {name sm})}
                   :encoding {:x {:field "category"
                                  :type "nominal"
                                  :axis {:labelLimit 0}}
                              :y {:field "category"
                                  :type "nominal"
                                  :axis {:labelLimit 0}}}}
        spec (medley/deep-merge base-spec
                                (heatmap-spec name)
                                (sort-spec (or sort-sm sm))
                                (when domain (domain-spec domain))
                                (when scheme (scheme-spec scheme)))]
    (json/write spec *out* :value-fn (fn [k v]
                                       (cond
                                         (and (number? v) (Double/isNaN v)) "NaN",
                                         (and (number? v) (Double/isInfinite v))
                                         (if (pos? v) "Infinity" "-Infinity"),
                                         :else v)))))
