using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UnityEditor.U2D.Common.Path
{
    internal enum ShapeType
    {
        Polygon,
        Spline
    }

    internal interface IShape
    {
        ShapeType type { get; }
        bool isOpenEnded { get; }
        ControlPoint[] ToControlPoints();
    }
}
