using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.U2D;
using Random = System.Random;


/// <summary>
/// Sample script to generate Custom Geometry. SpriteShapeColored genaretes Edges with Vertex Colors determined by the Gradient.
/// </summary>
[CreateAssetMenu(fileName = "SpriteShapeColored", menuName = "SpriteShapeCreators/SpriteShapeColored", order = 1)]
public class SpriteShapeColored : SpriteShapeGeometryCreator
{

    public float m_EdgeWidth = 1.0f;
    public Gradient m_Gradient = new Gradient();

    /// <summary>
    /// Set the version of Custom Geometry.
    /// </summary>
    /// <returns></returns>
    public override int GetVersion()
    {
        return 1;
    }

    /// <summary>
    /// Get Vertex Array Count. Allocate the maximum possible required.
    /// </summary>
    /// <param name="sc">SpriteShapeController for which the geometry is generated.</param>
    /// <returns></returns>
    public override int GetVertexArrayCount(SpriteShapeController sc)
    {
        return 32000;
    }

    static float2 BezierPoint(float2 st, float2 sp, float2 ep, float2 et, float t)
    {
        float2 xt = new float2(t);
        float2 nt = new float2(1.0f - t);
        float2 x3 = new float2(3.0f);
        return (sp * nt * nt * nt) + (st * nt * nt * xt * x3) + (et * nt * xt * xt * x3) + (ep * xt * xt * xt);
    }

    static float InterpolateLinear(float a, float b, float t)
    {
        return math.lerp(a, b, t);
    }

    static float InterpolateSmooth(float a, float b, float t)
    {
        float mu2 = (1.0f - math.cos(t * math.PI)) / 2.0f;
        return (a * (1 - mu2) + b * mu2);
    }

    static bool GenerateColumnsBi(float2 a, float2 b, float2 whsize, bool flip, ref float2 rt, ref float2 rb, float cph, float pivot)
    {
        float2 v1 = flip ? (a - b) : (b - a);
        if (math.length(v1) < 1e-30f)
            return false;

        float2 rvxy = new float2(-1f, 1f);
        float2 v2 = v1.yx * rvxy;
        float2 whsizey = new float2(whsize.y * cph);
        v2 = math.normalize(v2);

        float2 v3 = v2 * whsizey;
        rt = a - v3;
        rb = a + v3;

        float2 pivotSet = (rb - rt) * pivot;
        rt = rt + pivotSet;
        rb = rb + pivotSet;
        return true;
    }

    /// <summary>
    /// Function to generate the geometry.
    /// </summary>
    /// <param name="sc">SpriteShapeController for which the geometry is generated.</param>
    /// <param name="indices">Indices of the Triangle List.</param>
    /// <param name="positions">Positions of the Triangle List.</param>
    /// <param name="texCoords">TexCoords of the Triangle List.</param>
    /// <param name="tangents">Tangents of the Triangle List.</param>
    /// <param name="segments">Segments that define the Submeshes.</param>
    /// <param name="colliderData">ColliderData defines the Outline input to Collider.</param>
    /// <returns>JobHandle of the Job. Default if Inline.</returns>
    public override JobHandle MakeCreatorJob(SpriteShapeController sc,
        NativeArray<ushort> indices, NativeSlice<Vector3> positions, NativeSlice<Vector2> texCoords,
        NativeSlice<Vector4> tangents, NativeArray<UnityEngine.U2D.SpriteShapeSegment> segments, NativeArray<float2> colliderData)
    {
        NativeArray<Bounds> bounds = sc.spriteShapeRenderer.GetBounds();
        var spline = sc.spline;
        Bounds bds = new Bounds(spline.GetPosition(0), Vector3.zero);
        for (int i = 1; i < spline.GetPointCount(); ++i)
            bds.Encapsulate(spline.GetPosition(i));
        bds.extents = bds.extents * 1.4f; // Account for Edge Sprites.
        bounds[0] = bds;

        NativeSlice<Color32> clrArray;
        sc.spriteShapeRenderer.GetChannels(32 * 1024, out indices, out positions, out texCoords, out clrArray, out tangents);

        // Expand the Bezier.
        int ap = 0;
        int pc = spline.GetPointCount() - 1;
        float fmax = (float)(sc.splineDetail);
        NativeArray<float3> splinePos = new NativeArray<float3>( (pc * sc.splineDetail) + 1, Allocator.Temp);
        for (int i = 0; i < pc; ++i)
        {
            int j = i + 1;
            var cp = spline.GetPosition(i);
            var pp = spline.GetPosition(j);
            var smoothInterp = spline.GetTangentMode(i) == ShapeTangentMode.Continuous ||spline.GetTangentMode(j) == ShapeTangentMode.Continuous;

            float2 p0 = new float2(cp.x, cp.y);
            float2 p1 = new float2(pp.x, pp.y);
            float2 sp = p0;
            float2 rt = p0 + new float2(spline.GetRightTangent(i).x, spline.GetRightTangent(i).y);
            float2 lt = p1 + new float2(spline.GetLeftTangent(j).x, spline.GetLeftTangent(j).y);
            int cap = ap;
            float spd = 0, cpd = 0;

            for (int n = 0; n < sc.splineDetail; ++n)
            {
                var xp = splinePos[ap];
                float t = (float) n / fmax;
                float2 bp = BezierPoint(rt, p0, p1, lt, t);
                xp.x = bp.x;
                xp.y = bp.y;
                spd += math.distance(bp, sp);
                splinePos[ap++] = xp;
                sp = bp;
            }
            sp = p0;

            for (int n = 0; n < sc.splineDetail; ++n)
            {
                var xp = splinePos[cap];
                cpd += math.distance(xp.xy, sp);
                xp.z = smoothInterp ? InterpolateSmooth(spline.GetHeight(i), spline.GetHeight(j), cpd / spd) : InterpolateLinear(spline.GetHeight(i), spline.GetHeight(j), cpd / spd);
                splinePos[cap++] = xp;
                sp = xp.xy;
            }

        }

        var lp = splinePos[splinePos.Length - 1];
        lp = spline.GetPosition(pc);
        lp.z = spline.GetHeight(pc);
        splinePos[splinePos.Length - 1] = lp;

        int pointCount = (pc * sc.splineDetail) + 1, _v = 0, _i = 0;
        NativeArray<float2> upVerts = new NativeArray<float2>(pointCount, Allocator.Temp);
        NativeArray<float2> dnVerts = new NativeArray<float2>(pointCount, Allocator.Temp);

        for (int i = 0; i < pointCount; ++i)
        {
            var _l = (i == pointCount - 1);
            var _p= splinePos[i].xy;
            var _q= _l ? splinePos[i - 1].xy : splinePos[i + 1].xy;
            var _u= upVerts[i];
            var _d= dnVerts[i];
            GenerateColumnsBi(_p, _q, m_EdgeWidth, _l, ref _u, ref _d, splinePos[i].z, 0);
            upVerts[i] = _u;
            dnVerts[i] = _d;
            clrArray[_v] = m_Gradient.Evaluate((float) i / (float) pointCount);
            positions[_v++] = new Vector3(_u.x, _u.y, 0);
            clrArray[_v] = m_Gradient.Evaluate((float) i / (float) pointCount);
            positions[_v++] = new Vector3(_d.x, _d.y, 0);
        }

        for (int i = 0; i < _v - 2; i = i + 2)
        {
            indices[_i++] = (ushort)(i + 0);
            indices[_i++] = (ushort)(i + 1);
            indices[_i++] = (ushort)(i + 2);
            indices[_i++] = (ushort)(i + 1);
            indices[_i++] = (ushort)(i + 3);
            indices[_i++] = (ushort)(i + 2);
        }

        if (_v > 0)
        {
            for (int i = 0; i < _v; ++i)
                texCoords[i] = (positions[i] - bds.center) / (bds.size.x);

            var seg = segments[0];
            seg.geomIndex = 0;
            seg.indexCount = _i;
            seg.spriteIndex = 0;
            seg.vertexCount = _v;
            segments[0] = seg;

            seg.geomIndex = 0;
            seg.indexCount = 0;
            seg.spriteIndex = 0;
            seg.vertexCount = 0;
            for (int i = 1; i < segments.Length; ++i)
                segments[i] = seg;
        }

        return default(JobHandle);
    }

}