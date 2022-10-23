; this function calculates the average brightness in a small region to
; the left (i.e., with relative x-coordinate -1, and relative y-coordinates -1, 0, and 1) and
; the average brightness in a small region on the right side (with relative x-coordinate 1) of
; each pixel. If, at any particular pixel, the difference between these averages is greater than
; the specified threshold, then the pixel is marked with a one, meaning that it is an edge
; pixel. Otherwise it is marked with a 0. The threshold is multiplied by the overall average
; brightness, a process called "normalization." With normalization, the threshold adapts to
; the image, becoming small in regions where the image is generally dark, and large where
; the image is generally bright.

(*defun find-edges-between-left-and-right!! (brightness-pvar threshold)
  (*let* ((average-brightness-on-the-left
            (/!! (+!! (pref-grid-relative!! brightness-pvar (!! -1) (!! -1))
                      (pref-grid-relative!! brightness-pvar (!! -1) (!!  0))
                      (pref-grid-relative!! brightness-pvar (!! -1) (!!  1)))
                 (!! 3.0)))
          (average-brightness-on-the-right
            (/!! (+!! (pref-grid-relative!! brightness-pvar (!!  1) (!! -1))
                      (pref-grid-relative!! brightness-pvar (!!  1) (!!  0))
                      (pref-grid-relative!! brightness-pvar (!!  1) (!!  1)))
                 (!! 3.0)))
          (average-brightness-overall
            (/!! (+!! average-brightness-on-the-left
                      average-brightness-on-the-right)
                 (!! 2.0))))
         (if!! (>!! (absolute-value!! (-!! average-brightness-on-the-left
                                           average-brightness-on-the-right))
                    (*!! (!! threshold) average-brightness-overall))
               (!! 1)
               (!! 0))))

; Since this program compares regions on the left and right sides of a pixel, it works only
; for edges that are more or less vertical. It is easy to write a program that finds horizontal
; edges by having it compare small regions on the top and bottom of a pixel, in the same way
; that this program compares regions on the left and right. The same could be done edges
; in both diagonal directions. The four programs may then be combined to find all edges in
; the following way:

(*defun find-all-edges!! (brightness-pvar threshold)
  (if!! (or!! (=! (!! 1) (find-edges-between-left-and-right!!              brightness-pvar threshold))
              (=! (!! 1) (find-edges-between-above-and-below!!             brightness-pvar threshold))
              (=! (!! 1) (find-edges-between-upper-left-and-lower-right!!  brightness-pvar threshold))
              (=! (!! 1) (find-edges-between-lower-left-and-upper-right!!  brightness-pvar threshold)))
        (!! 1)
        (!! 0)))

;;; this is just a regular lisp array, but each element of this array will be a pvar.
;;; notice that we'll try to find positional differences of up to 30 pixels. (note: each
;;; one of the pvars in this array will hold an "alignment-table-slot for every pixel,
;;; as discussed in the text).
(defvar *array-of-pvars-holding-matches-at-each-shift* (make-array 30))

(*defun fillup-pvars-wherever-edges-align (left-edges right-edges)
  ;; this function records the edge-pixel match-ups at every shift;
  ;; this is, this program creates "match-up images", as shown in figure 4.4.
  (dotimes (i 30)
    (aset (if!! (=!! left-edges
                     (pref-grid-relative!! right-edges (!! i) (!! 0)))
                        ; ^ this PREF-GRID-RELATIVE!! accomplishes the "sliding" process.
                (!! 1)
                (!! 0))
          *array-of-pvars-holding-matches-at-each-shift*
          i)))

;;; The next step in the process is to decide at each pixel position which shift produced
;;; the best match-up. Most locations will contain a somewhat random pattern of match-up
;;; pixels. However, at some locations, the local neighborhood of match-ups will be very dense
;;; and regular, indicating that the shift responsible for that match-up image is probably the
;;; correct shift for that neighborhood.
;;; The following *Lisp program measures the density or alignment quality of every neigh-
;;; borhood. It does so by, counting the number of 1’s (match-ups) in a square around each
;;; pixel. The counting process is accomplished in parallel, for all pixels at once, on the Con-
;;; nection Machine system.

;;; the square for each pixel is to be centered on that pixel. Because a DOTIMES loop always
;;; produces values starting at zero, it is necessary to subtract one-half the width of the
;;; square from the loop variable in order to get relative indexes that are centered on zero.

(*defun add-up-all-pixels-in-a-square (pvar width-of-square)
  (let ((one-half-the-square-width (/ width-of-square 2)))
    (*let ((total (!! 0)))
      (dotimes (relative-x width-of-square)
        (dotimes (relative-y width-of-square)
          (*set total
                (+!! total
                     (pred-grid-relative!!
                       pvar
                       (- relative-x one-half-the-square-width)
                       (- relative-y one-half-the-square-width))))))
      total)))

(defvar *array-of-pvars-holding-scores-at-each-shift* (make-array 30))
  ;;; another lisp array holding pvars.

(*defun fillup-pvars-with-match-scores (width-of-square)
  ;; WIDTH-OF-SQUARE will typically be 21
  (dotimes (i 30)
    (*let ((sum-of-all-nearby-pixels
            (add-up-all-pixels-in-a-square
              (aref *array-of-pvars-holding-matches-at-each-shift* i)
              width-of-square)))
      (*if (=!! (aref *array-of-pvars-holding-matches-at-each-shift* i)
                (!! 1)) ;;; record a score wherever there was a match-up
           (*set sum-of-all-nearby-pixels
                 *array-of-pvars-holding-scores-at-each-shift*
                 i)))))

;;; this function computes the web of known shifts. recall that
;;; the shift at each pixel corresponds directly to the elevation.

(*defun find-the-shifts-of-the-highest-scoring-matches ()
  (*let ((best-scores (!! 0))
        (winning-shifts (!! 0)))
    ;; the following DOTIMES loop makes sure that each pixel in the BEST-SCORES pvar
    ;; contains the maximum score found at any shift.
    (dotimes (i 30)
      (*if (>!! (aref *array-of-pvars-holding-scores-at-each-shift* i)
                best-scores)
           (*set best-scores
                 (aref *array-of-pvars-holding-scores-at-each-shift* i))))
    ;; the following DOTIMES loop records a "winning" shift at every pixel whose score
    ;; is the best.
    (dotimes (i 30)
      (*if (=!! (aref *array-of-pvars-holding-scores-at-each-shift* i) best-scores)
           (*set winning-shifts (!! (1+ i)))))
    winning-shifts))

(defun fill-in-web-holes (web-of-known-elevations times-to-repeat)
  ;; each time though the loop, every pixel not on the web (i.e., every pixel that is not
  ;; zero to begin with) takes on the average elevation of its four neighbors. therefore,
  ;; the web pixels gradually "spread" their elevations across the holes, while they
  ;; themselves remain unchanged.
  (dotimes (i times-to-repeat)
    (*let ((not-fixed (zerop!! web-of-known-elevations)))
      (*if not-fixed
        (*set web-of-known-elevations
              (/! (+!!
                    (pref-grid-relative!! web-of-known-elevations (!!  1) (!!  0))  ; neighbor to the right
                    (pref-grid-relative!! web-of-known-elevations (!!  0) (!!  1))  ; neighbor above
                    (pref-grid-relative!! web-of-known-elevations (!! -1) (!!  0))  ; neighbor to the left
                    (pref-grid-relative!! web-of-known-elevations (!!  0) (!! -1))) ; neighbot below
                  (!! 4))))))
  web-of-known-elevations)  ;;; this is now a more or less smooth surface.

(defun draw-contour-map (number-of-contour-lines pvar-of-smooth-continuous-elevations)
  ;; the idea is to divide the whole range of elevations into a number of intervals,
  ;; then to draw a contour line at every interval.
  (let ((max-elevation (*max pvar-of-smooth-continuous-elevations))
        (min-elevation (*min pvar-of-smooth-continuous-elevations))
        (range-of-elevations (- max-elevation min-elevation))
        (contour-line-interval (/ range-of-elevations number-of-contour-lines)))
     ;; now the variable CONTOUR-LINE-INTERVAL tells us how many elevations, or shifts,
     ;; to skip between contour lines.
    (if!! (zerop!! (mod!! (-!! pvar-of-smooth-continuous-elevations (!! min-elevation))
                          (!! contour-line-interval)))
          (!! 1)        ;; this if!! draws all the elevation contours
          (!! 0))))     ;; at once, returning a bit map suitable for immediate display.

