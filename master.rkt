#lang racket

(require 2htdp/batch-io)

(require "decision_functions.rkt")

;input dataset
(provide toytrain)
(define toytrain "../data/toy_train.csv")

(provide titanictrain)
(define titanictrain "../data/titanic_train.csv")

(provide mushroomtrain)
(define mushroomtrain "../data/mushrooms_train.csv")

;output tree (dot file)
(provide toyout)
(define toyout "../output/toy-decision-tree.dot")

(provide titanicout)
(define titanicout "../output/titanic-decision-tree.dot")

(provide mushroomout)
(define mushroomout "../output/mushroom-decision-tree.dot")

;reading input datasets
;read the csv file myfile as a list of strings
;with each line of the original file as an element of the list
;further split each line at commas
;so then we have a list of list of strings
(provide toy-raw)
(define toy-raw (cdr (read-csv-file toytrain)))

(provide titanic-raw)
(define titanic-raw (map cddr (cdr (read-csv-file titanictrain))))

(provide mushroom-raw)
(define mushroom-raw (cdr (read-csv-file mushroomtrain)))

;function to convert data to internal numerical format
;(features . result)
(provide format)
(define (format data)
      (cons (map string->number (cdr data)) (string->number (car data))))

;list of (features . result)
(provide toy)
(define toy (map format toy-raw))

(provide titanic)
(define titanic (map format titanic-raw))

(provide mushroom)
(define mushroom (map format mushroom-raw))

;============================================================================================================
;============================================================================================================
;============================================================================================================

;get fraction of result fields that are 1
;used to find probability value at leaf
(provide get-leaf-prob)
(define (get-leaf-prob data)
  (if (null? data) 0
  (/ (count (lambda (x) (= 1 (cdr x))) data) (length data))
  ))

;get entropy of dataset
(provide get-entropy)
(define (get-entropy data)
  (define p (get-leaf-prob data))
  (define n (- 1 p))
  (if (or (zero? n) (zero? p)) 0
  (+ (* (- p) (log p 2)) (* (- n) (log n 2)))
  ))

;find the difference in entropy achieved
;by applying a decision function f to the data
(provide entropy-diff)
(define (entropy-diff f data)
  (define f-feature-target (map (lambda (x) (cons (f (car x)) (cdr x))) data))
  (define num_subsets (+ (apply max (map car f-feature-target)) 1))
  (define data_size (length data))
  (define cl-data (classify f-feature-target num_subsets))
  (define exp-entropy (/ (apply + (map (lambda (x) (* (length x) (get-entropy x))) cl-data)) data_size))
  (- (get-entropy data) exp-entropy)
)

(define (classify f-t num_subsets)
  (define subset-list (make-list num_subsets '()))
  (define (classify-helper f-t lst)
    (if (null? f-t) lst
                    (let ((pos (caar f-t)))
                    (classify-helper (cdr f-t) (list-update lst pos (lambda (x) (cons (car f-t) x)))))))
  (classify-helper f-t subset-list))

;choose the decision function that most reduces entropy of the data
(define (eff-entropy-diff f data)
  (define f-feature-target (map (lambda (x) (cons (f (car x)) (cdr x))) data))
  (define num_subsets (+ (apply max (map car f-feature-target)) 1))
  (define data_size (length data))
  (define cl-data (classify f-feature-target num_subsets))
  (define exp-entropy (/ (apply + (map (lambda (x) (* (length x) (get-entropy x))) cl-data)) data_size))
  (- exp-entropy))
  
(provide choose-f)
(define (choose-f candidates data) ; returns a decision function
    (define (choose-f-helper candidates data present_max-f)
         (if (null? candidates) (car present_max-f)
                                (let ((ed (eff-entropy-diff (cdar candidates) data)))
                                     (choose-f-helper (cdr candidates) data (if (< (cdr present_max-f) ed) (cons (car candidates) ed)
                                                                                                           present_max-f)))))
    (choose-f-helper candidates data (cons -inf.0 -inf.0)))

(provide DTree
         DTree-kids)
(struct DTree (desc func kids))

;build a decision tree (depth limited) from the candidate decision functions and data
(define (classify-2 opt-f data)
  (define f-data (map (lambda (x) (opt-f (car x))) data))
  (define num_subsets (+ (apply max f-data) 1))
  (define subset-list (make-list num_subsets '()))
  (define (classify-helper data lst)
    (if (null? data) lst
                    (let ((pos (opt-f (caar data))))
                    (classify-helper (cdr data) (list-update lst pos (lambda (x) (cons (car data) x)))))))
  (classify-helper data subset-list))

(provide build-tree)
(define (build-tree candidates data depth) 
  (cond [(not (null? data))
         (let* ((opt-f (choose-f candidates data))
        (new_candidates (remove opt-f candidates)))
    (if (or (= 1 depth) (null? new_candidates)) (DTree (car opt-f) (cdr opt-f) (map (λ (y) (DTree (~a (get-leaf-prob y)) "" '())) (classify-2 (cdr opt-f) data))) 
                    (DTree (car opt-f) (cdr opt-f) (map (λ (y) (build-tree new_candidates y (- depth 1))) (classify-2 (cdr opt-f) data))))  
  )]
  [(null? data) (DTree (~a 0) "" '())]))

;given a test data (features only), make a decision according to a decision tree
;returns probability of the test data being classified as 1
(provide make-decision)
(define (make-decision tree test)
   (mk-dec tree test)
  )

(define (mk-dec tree sample)
  (if (null? (DTree-kids tree)) (string->number (DTree-desc tree))
                                (let ((pos ((DTree-func tree) sample)))
                                      (mk-dec (list-ref (DTree-kids tree) pos) sample))))

;============================================================================================================
;============================================================================================================
;============================================================================================================

;annotate list with indices
(define (pair-idx lst n)
  (if (empty? lst) `() (cons (cons (car lst) n) (pair-idx (cdr lst) (+ n 1))))
  )

;generate tree edges (parent to child) and recurse to generate sub trees
(define (dot-child children prefix tabs)
  (apply string-append
         (map (lambda (t)
                (string-append tabs
                               "r" prefix
                               "--"
                               "r" prefix "t" (~a (cdr t))
                               "[label=\"" (~a (cdr t)) "\"];" "\n"
                               (dot-helper (car t)
                                           (string-append prefix "t" (~a (cdr t)))
                                           (string-append tabs "\t")
                                           )
                               )
                ) children
                  )
         )
  )

;generate tree nodes and call function to generate edges
(define (dot-helper tree prefix tabs)
  (let* ([node (match tree [(DTree d f c) (cons d c)])]
         [d (car node)]
         [c (cdr node)])
    (string-append tabs
                   "r"
                   prefix
                   "[label=\"" d "\"];" "\n\n"
                   (dot-child (pair-idx c 0) prefix tabs)
                   )
    )
  )

;output tree (dot file)
(provide display-tree)
(define (display-tree tree outfile)
  (write-file outfile (string-append "graph \"decision-tree\" {" "\n"
                                     (dot-helper tree "" "\t")
                                     "}"
                                     )
              )
  )
;============================================================================================================
;============================================================================================================
;============================================================================================================
(define a (classify-2 (cdr y3) toy))