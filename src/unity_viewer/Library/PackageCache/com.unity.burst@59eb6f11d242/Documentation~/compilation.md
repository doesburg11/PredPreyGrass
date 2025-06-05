# Burst compilation

Use Burst's custom C# attributes to define which parts of your code Burst compiles. These attributes and their parameters also allow you to configure a range of compilation options to improve Burst performance in different contexts.

|**Topic**|**Description**|
|---|---|
|[Marking code for Burst compilation](compilation-burstcompile.md)|Use the `[BurstCompile]` attribute to mark code for Burst compilation. Use attribute paraeters to customize aspects of Burst compilation and improve performance. |
|[Excluding code from Burst compilation](compilation-burstdiscard.md)|Use the `[BurstDiscard]` attribute to selectively exclude portions of code from Burst compilation.|
|[Defining Burst options for an assembly](compilation-burstdiscard.md)|Apply the `[BurstCompile]` attribute at the assembly level to define Burst compilation options for a whole assembly.|
|[Burst compilation in Play mode](compilation-synchronous.md)|Burst provides the option to compile asynchronously or synchronously in Play mode. Understand these options and how and when to configure synchronous compilation.|
|[Generic jobs](compilation-generic-jobs.md)|Understand important limitations in Burst's support for generic jobs and function pointers.|
|[Compilation warnings](compilation-warnings.md)|Fix common compilation warnings.|

## Additional resources

* [C# language support](csharp-language-support.md)