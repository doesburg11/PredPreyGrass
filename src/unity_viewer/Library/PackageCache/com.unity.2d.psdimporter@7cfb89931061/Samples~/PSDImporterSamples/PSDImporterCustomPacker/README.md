# PSDImporter Custom Image Packer

This example shows how to override the default image packing algorithm in the PSDImporter.

The example utilizes the `m_Pipeline` SerializedProperty that is defined in the PSDImporter.

The `m_Pipeline` is a ScriptableObject reference and in the PSDImporter it will determine what method is available in the SciptableObject and execute those methods accordingly.

Refer to the `CustomPackScriptableObject.cs` for more details.