using System;
using Unity.Collections;
using Unity.Mathematics;

namespace UnityEngine.U2D
{

    /// <summary>
    /// SpriteShape Object placement Modes.
    /// </summary>
    public enum SpriteShapeObjectPlacementMode
    {
        /// <summary>
        /// Allows editing the transform of the Object while keeping it on the surface of the spline.
        /// </summary>
        Auto,
        /// <summary>
        /// Allows movement strictly with the Ratio and Start, End points.
        /// </summary>
        Manual
    };

    /// <summary>
    /// SpriteShapeObjectPlacement helps placing a game Object along the Spline.
    /// </summary>
    [ExecuteInEditMode]
    [ExecuteAlways]
    public class SpriteShapeObjectPlacement : MonoBehaviour
    {

        [SerializeField]
        SpriteShapeController m_SpriteShapeController;

        [SerializeField]
        bool m_SetNormal = true;

        [SerializeField]
        SpriteShapeObjectPlacementMode m_Mode;

        [SerializeField]
        [Min(0)]
        int m_StartPoint = 0;

        [SerializeField]
        [Min(0)]
        int m_EndPoint = 1;

        [SerializeField]
        float m_Ratio = 0.5f;

        /// Internal Hash Code.
        int m_ActiveHashCode = 0;
        static readonly float kMaxDistance = 10000.0f;
        private static readonly int kMaxIteration = 128;

        /// <summary>
        /// Dictates whether the object needs to be rotated to the normal along the spline.
        /// </summary>
        public bool setNormal
        {
            get { return m_SetNormal; }
            set { m_SetNormal = value; }
        }

        /// <summary>
        /// Set SpriteShape Object placement mode.
        /// </summary>
        public SpriteShapeObjectPlacementMode mode
        {
            get { return m_Mode; }
            set { m_Mode = value; }
        }

        /// <summary>
        /// Ratio of the distance between the startPoint and the endPoint. Must be between 0 and 1
        /// </summary>
        public float ratio
        {
            get { return m_Ratio; }
            set { m_Ratio = value; }
        }

        /// <summary>
        /// Source SpriteShapeController gameObject that contains the Spline to which this Object is placed along.
        /// </summary>
        public SpriteShapeController spriteShapeController
        {
            get { return m_SpriteShapeController; }
            set { m_SpriteShapeController = value; }
        }

        /// <summary>
        /// Start point of the pair of points between which the object is placed. Must be between 0 and points in Spline.
        /// </summary>
        public int startPoint
        {
            get { return m_StartPoint; }
            set { m_StartPoint = value; }
        }

        /// <summary>
        /// End point of the pair of points between which the object is placed. Must be between 0 and points in Spline and larger than StartPoint.
        /// </summary>
        public int endPoint
        {
            get { return m_EndPoint; }
            set { m_EndPoint = value; }
        }

        bool PlaceObjectOnHashChange()
        {
            if (null == spriteShapeController)
                return false;

            unchecked
            {
                var count = 0;
                int spriteShapeHashCode = (int) 2166136261 ^ spriteShapeController.splineHashCode;
                spriteShapeHashCode = spriteShapeHashCode * 16777619 ^ spriteShapeController.spriteShapeHashCode;
                var ssTransform = spriteShapeController.gameObject.transform;
                var pos = gameObject.transform.position;
                var rot = gameObject.transform.rotation;
                spriteShapeHashCode = spriteShapeHashCode * 16777619 ^ (setNormal ? 1 : 0);
                spriteShapeHashCode = spriteShapeHashCode * 16777619 ^ (startPoint);
                spriteShapeHashCode = spriteShapeHashCode * 16777619 ^ (endPoint);
                spriteShapeHashCode = spriteShapeHashCode * 16777619 ^ ssTransform.position.GetHashCode();
                spriteShapeHashCode = spriteShapeHashCode * 16777619 ^ ssTransform.rotation.GetHashCode();
                do
                {
                    // SpriteShape.
                    // Local Stuff.
                    int hashCode = spriteShapeHashCode * 16777619 ^ Math.Round(pos.x * 1000.0f).GetHashCode();
                    hashCode = hashCode * 16777619 ^ Math.Round(pos.y * 1000.0f).GetHashCode();
                    hashCode = hashCode * 16777619 ^ Math.Round(pos.z * 1000.0f).GetHashCode();
                    hashCode = hashCode * 16777619 ^ Math.Round(ratio * 1000.0f).GetHashCode();
                    hashCode = hashCode * 16777619 ^ Math.Round(rot.x * 1000.0f).GetHashCode();
                    hashCode = hashCode * 16777619 ^ Math.Round(rot.y * 1000.0f).GetHashCode();
                    hashCode = hashCode * 16777619 ^ Math.Round(rot.z * 1000.0f).GetHashCode();
                    hashCode = hashCode * 16777619 ^ Math.Round(rot.w * 1000.0f).GetHashCode();

                    if (m_ActiveHashCode != hashCode)
                    {
                        var run = Place();
                        m_ActiveHashCode = hashCode;
                        if (!run)
                            break;
                    }
                    else
                    {
                        break;
                    }
                } while (count++ < kMaxIteration);
            }

            return false;
        }

        static float Angle(Vector3 a, Vector3 b)
        {
            float dot = Vector3.Dot(a, b);
            float det = (a.x * b.y) - (b.x * a.y);
            return Mathf.Atan2(det, dot) * Mathf.Rad2Deg;
        }
        float GetDistance(float dist, int spoint, int epoint, ref int start, ref int end, ref float r, NativeArray<ShapeControlPoint> shapePoints)
        {
            start = -1;
            var detail = spriteShapeController.splineDetail;
            var distance = 0.0f;
            var division = (float)(detail - 1);
            var pointCount = shapePoints.Length;

            for (int i = spoint; i < epoint; ++i)
            {
                var j = i + 1;
                if (j == pointCount) j = 0;
                var cp = shapePoints[i];
                var pp = shapePoints[j];

                var p0 = cp.position;
                var p1 = pp.position;
                var sp = p0;
                var rt = p0 + cp.rightTangent;
                var lt = p1 + pp.leftTangent;
                var ld = 0.0f;
                var pd = 0.0f;
                var st = false;

                if (dist != 0 && dist > distance)
                {
                    start = i;
                    end = (i + 1 == pointCount) ? 0 : (i + 1);
                    pd = distance;
                    st = true;
                }

                for (int n = 1; n < detail; ++n)
                {
                    var t = (float)n / division;
                    var bp = BezierUtility.BezierPoint(rt, p0, p1, lt, t);
                    var d = math.distance(bp, sp);
                    ld += d;
                    distance += d;
                }

                if (st)
                {
                    var diff = dist - pd;
                    r = diff / ld;
                }
            }
            return distance;
        }

        Vector3 PlaceObjectInternal(int sp, int ep, float t, NativeArray<ShapeControlPoint> shapePoints)
        {
            ep = ep % shapePoints.Length;
            var p0 = shapePoints[sp].position;
            var p1 = shapePoints[ep].position;
            var rt = p0 + shapePoints[sp].rightTangent;
            var lt = p1 + shapePoints[ep].leftTangent;
            var bp = BezierUtility.BezierPoint(rt, p0, p1, lt, t);
            var position = new Vector3(bp.x, bp.y, 0);
            var srcTransform = spriteShapeController.transform.localToWorldMatrix;
            var dstTransform = gameObject.transform;
            var p = srcTransform.MultiplyPoint3x4(position);
            var d = dstTransform.position;
            if (m_Mode == SpriteShapeObjectPlacementMode.Auto)
                d.y = p.y;
            else
                d = p;
            dstTransform.position = d;

            if (setNormal)
            {
                var _r = math.clamp(t, 0.002f, 0.998f);
                var pp = BezierUtility.BezierPoint(rt, p0, p1, lt, _r - 0.001f);
                bp = BezierUtility.BezierPoint(rt, p0, p1, lt, _r);
                var np = BezierUtility.BezierPoint(rt, p0, p1, lt, _r + 0.001f);
                var _lt = Vector3.Normalize(new Vector3(pp.x, pp.y, 0) - new Vector3(bp.x, bp.y, 0));
                var _rt = Vector3.Normalize(new Vector3(np.x, np.y, 0) - new Vector3(bp.x, bp.y, 0));
                var a = Angle(Vector3.up, _lt);
                var b = Angle(_lt, _rt);
                var c = a + (b * 0.5f);
                if (b > 0)
                    c = (180 + c);
                var rotation = Quaternion.Euler(0, 0, c);
                dstTransform.rotation = srcTransform.rotation * rotation;
            }
            return d;
        }

        Vector3 PlaceObject(Spline spline, int sp, int ep, ref bool run)
        {
            var shapePoints = spriteShapeController.GetShapeControlPoints();
            if (sp > shapePoints.Length || ep > shapePoints.Length)
            {
                run = false;
                return Vector3.zero;
            }
                
            var t = math.clamp(ratio, 0.0001f, 0.9999f);
            if (ep - sp == 1)
            {
                run = true;
                return PlaceObjectInternal(sp, ep, t, shapePoints);
            }
            else
            {
                var s = 0;
                var e = 0;
                var d = 0.0f;
                var r = 0.0f;
                d = GetDistance(d, sp, ep, ref s, ref e, ref r, shapePoints) * t;
                GetDistance(d, sp, ep, ref s, ref e, ref r, shapePoints);
                if (s >= 0)
                {
                    run = true;
                    return PlaceObjectInternal(s, e, r, shapePoints);
                }
            }
            run = false;
            return Vector3.zero;
        }

        int GetSplinePointCount()
        {
            var spline = spriteShapeController.spline;
            var pointCount = spline.GetPointCount();
            pointCount = spline.isOpenEnded ? pointCount - 1 : pointCount;
            return pointCount;
        }

        bool Place()
        {
            var pointCount = GetSplinePointCount();
            var run = false;
            if (m_Mode == SpriteShapeObjectPlacementMode.Manual)
            {
                var sp = math.clamp(startPoint, 0, pointCount);
                var ep = math.clamp(endPoint, 0, pointCount);
                if (sp >= ep)
                {
                    endPoint = pointCount;
                    Debug.LogWarning("Invalid End point and it has been clamped", transform);
                }
                PlaceObject(spriteShapeController.spline, sp, ep, ref run);
                return run;
            }

            var distance = kMaxDistance;
            var position = transform.position;
            var closestPoint = Vector3.zero;
            {
                int tp = 0, np = 0;
                float et = 100, dist = kMaxDistance;
                var spline = spriteShapeController.spline;
                var matrix = spriteShapeController.transform.localToWorldMatrix;
                var splinePointCount = spline.GetPointCount();                

                for (int i = 0; i < pointCount; ++i)
                {
                    var ni = (i + 1) % spline.GetPointCount();
                    var thisposition = matrix.MultiplyPoint3x4(spline.GetPosition(i));
                    var nextPosition = matrix.MultiplyPoint3x4(spline.GetPosition(ni));
                    var rightTangent = spline.GetRightTangent(i) + thisposition;
                    var leftTangent = spline.GetLeftTangent(ni) + nextPosition;

                    float t;
                    closestPoint = BezierUtility.ClosestPointOnCurve(
                        position,
                        thisposition,
                        nextPosition,
                        rightTangent,
                        leftTangent,
                        0.0001f,
                        out t);
                    float _d = (closestPoint - position).magnitude;
                    if (_d < dist)
                    {
                        tp = i;
                        np = ni;
                        et = t;
                        dist = _d;
                    }

                }

                if (tp >= 0 && tp < splinePointCount && np >= 0 && np < splinePointCount)
                {
                    startPoint = tp;
                    endPoint = np == 0 ? tp + 1 : np;
                    ratio = et;
                    position = PlaceObject(spline, startPoint, endPoint, ref run);
                }
            }
            return run;
        }

        void Start ()
        {
            PlaceObjectOnHashChange();
        }

        void Update ()
        {
            PlaceObjectOnHashChange();
        }

    }

}