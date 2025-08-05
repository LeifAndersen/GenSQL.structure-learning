(ns gensql.structure-learning.qc.splom
  "Code related to producing a vega-lite spec for a scatter plot matrix."
  (:require [clojure.data.json :as json]
            [clojure.edn :as edn]
            [cheshire.core :as cheshire]
            [cheshire.factory :as factory]
            [gensql.structure-learning.dvc :as dvc]
            [gensql.structure-learning.qc.vega :as vega]
            [gensql.structure-learning.qc.util :refer [filtering-summary should-bin? bind-to-element
                                                       obs-data-color synthetic-data-color
                                                       unselected-color vega-type-fn
                                                       vl5-schema]]))

(defn scatter-plot-for-splom [col-1 col-2 vega-type correlation samples]
  (let [f-sum (filtering-summary [col-1 col-2] vega-type nil samples)
        base-spec {:layer [{:mark {:type "point"
                                   :tooltip {:content "data"}
                                   :filled true
                                   :size {:expr "splomPointSize"}}
                            :params [{:name "zoom-control-splom"
                                      :bind "scales"
                                      :select {:type "interval"
                                               :resolve "global"
                                               :on "[mousedown[event.shiftKey], window:mouseup] > window:mousemove!",
                                               :translate "[mousedown[event.shiftKey], window:mouseup] > window:mousemove!",
                                               :clear "dblclick[event.shiftKey]"
                                               :zoom "wheel![event.shiftKey]"}}
                                     {:name "brush-all"
                                      :select {:type "interval"
                                               :resolve "global"
                                               :on "[mousedown[!event.shiftKey], window:mouseup] > window:mousemove"
                                               :translate "[mousedown[!event.shiftKey], window:mouseup] > window:mousemove!",
                                               :clear "dblclick[!event.shiftKey]"
                                               :zoom "wheel![!event.shiftKey]"}}]
                            :transform [{:window [{:op "row_number", :as "row_number_subplot"}]
                                         :groupby ["collection"]}
                                        {:filter {:or [{:and [{:field "collection" :equal "observed"}
                                                              {:field "row_number_subplot" :lte {:expr "numObservedPoints"}}]}
                                                       {:and [{:field "collection" :equal "synthetic"}
                                                              {:field "row_number_subplot" :lte (:num-valid f-sum)}
                                                              {:field "row_number_subplot" :lte {:expr "numVirtualPoints"}}]}]}}]
                            :encoding {:x {:field col-1
                                           :type "quantitative"
                                           :axis {:gridOpacity 0.4
                                                  :title col-1}}
                                       :y {:field col-2
                                           :type "quantitative"
                                           :axis {:gridOpacity 0.4
                                                  :title col-2}}
                                       :opacity {:field "collection"
                                                 :scale {:domain ["observed", "synthetic"]
                                                         :range [{:expr "splomAlphaObserved"} {:expr "splomAlphaVirtual"}]}
                                                 :legend nil}
                                       :color {:condition {:param "brush-all"
                                                           :field "collection"
                                                           :scale {:domain ["observed", "synthetic"]
                                                                   :range [obs-data-color synthetic-data-color]}
                                                           :legend {:orient "top"
                                                                    :title nil
                                                                    :offset 30}}
                                               :value unselected-color}}}]}]
    (vega/regression-line col-1 col-2 correlation samples base-spec)))

(defn layered-histogram [col vega-type samples]
  (let [col-type (vega-type col)
        bin-flag (should-bin? col-type)
        f-sum (filtering-summary [col] vega-type nil samples)]
    {:params [{:name "brush-all"
               :select {:type "interval" :encodings ["x"]}}]
     :transform [{:window [{:op "row_number", :as "row_number_subplot"}]
                  :groupby ["collection"]}
                 {:filter {:or [{:and [{:field "collection" :equal "observed"}
                                       {:field "row_number_subplot" :lte {:expr "numObservedPoints"}}]}
                                {:and [{:field "collection" :equal "synthetic"}
                                       {:field "row_number_subplot" :lte (:num-valid f-sum)}
                                       {:field "row_number_subplot" :lte {:expr "numVirtualPoints"}}]}]}}]
     :mark {:type "bar"
            :tooltip {:content "data"}}
     :encoding {:x {:bin bin-flag
                    :field col
                    :type col-type}
                :y {:aggregate "count"
                    :type "quantitative"
                    :stack nil}
                :opacity {:field "collection"
                          :scale {:domain ["observed", "synthetic"]
                                  :range [{:expr "histAlphaObserved"} {:expr "histAlphaVirtual"}]}
                          :legend nil}
                :color {:field "collection"
                        :scale {:domain ["observed" "synthetic"]
                                :range [obs-data-color synthetic-data-color]}
                        :legend {:orient "top"
                                 :offset 30
                                 :title nil}}}}))

(defn spec
  "Produces a vega-lite spec for the QC SPLOM app.
  Paths to samples and schema are required.
  Path to correlation data is optional."
  [{sample-path :samples schema-path :schema correlation-path :correlation}]
  (let [schema (-> schema-path str slurp edn/read-string)
        samples (-> sample-path str slurp edn/read-string)
        correlation
        (binding [factory/*json-factory* (factory/make-json-factory
                                          {:allow-non-numeric-numbers true})]
          (some-> correlation-path str slurp (cheshire/parse-string true)))

        ;; Visualize the columns set in params.yaml.
        ;; If not specified, visualize all the columns.
        cols (or (get-in (dvc/yaml) [:qc :columns])
                 (keys schema))

        vega-type (vega-type-fn schema)
        cols (->> cols
                  (map keyword)
                  ;; Get just the quantitative columns.
                  (filter (comp #{"quantitative"} vega-type))
                  ;; We will visualize at most 8 columns.
                  (take 8))

        plot-rows (for [col-1 cols]
                    (let [specs (for [col-2 cols]
                                  (if (= col-1 col-2)
                                    (layered-histogram col-2 vega-type samples)
                                    ;; Each column has the same x-dimension.
                                    (scatter-plot-for-splom col-1 col-2 vega-type
                                                            correlation samples)))]
                      {:vconcat specs
                       :spacing 80
                       :bounds "flush"
                       :resolve {:legend {:color "shared"}
                                 :scale {:color "shared"
                                         :opacity "independent"
                                         :x "shared"}}}))

        num-obs-samples (-> (group-by :collection samples)
                            (get "observed")
                            (count))
        num-synthetic-samples (-> (group-by :collection samples)
                                (get "synthetic")
                                (count))

        spec {:$schema vl5-schema
              :params [{:name "histAlphaObserved"
                        :value 0.7
                        :bind {:input "range" :min 0 :max 1 :step 0.05
                               :name "histograms - alpha (observed data)"}}
                       {:name "histAlphaVirtual"
                        :value 0.7
                        :bind {:input "range" :min 0 :max 1 :step 0.05
                               :name "histograms - alpha (synthetic data)"}}
                       {:name "splomAlphaObserved"
                        :value 0.7
                        :bind {:input "range" :min 0 :max 1 :step 0.05
                               :name "scatter plots - alpha (observed data)"}}
                       {:name "splomAlphaVirtual"
                        :value 0.7
                        :bind {:input "range" :min 0 :max 1 :step 0.05
                               :name "scatter plots - alpha (synthetic data)"}}
                       {:name "splomPointSize"
                        :value 30
                        :bind {:input "range" :min 1 :max 100 :step 1
                               :name "scatter plots - point size"}}
                       {:name "numObservedPoints"
                        :value num-obs-samples
                        :bind {:input "range" :min 1 :max num-obs-samples :step 1
                               :name "number of points (observed data)"}}
                       {:name "numVirtualPoints"
                        :value num-synthetic-samples
                        :bind {:input "range" :min 1 :max num-synthetic-samples :step 1
                               :name "number of points (synthetic data)"}}
                       {:name "showRegression"
                        :value false
                        :bind {:input "checkbox"
                               :name "regression lines"}}]
              :data {:values samples}
              :transform [{:window [{:op "row_number", :as "row_number"}]
                           :groupby ["collection"]}]
              :hconcat plot-rows
              :spacing 40
              :config {:countTitle "Count"}
              :resolve {:legend {:color "shared"}
                        :scale {:color "shared"
                                :opacity "independent"
                                :x "independent"}}}]
    (-> spec
        (update :params bind-to-element "#controls")
        (json/pprint))))
