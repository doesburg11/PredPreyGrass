using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    internal interface IWeightsGenerator
    {
        BoneWeight[] Calculate(string name, in float2[] vertices, in int[] indices, in int2[] edges, in float2[] controlPoints, in int2[] bones, in int[] pins);
        JobHandle CalculateJob(string name, in float2[] vertices, in int[] indices, in int2[] edges, in float2[] controlPoints, in int2[] bones, in int[] pins, SpriteJobData sd);
    }
}
