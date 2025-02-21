#include <gtest/gtest.h>
#include "LinearLayer.h"
#include "Tensor4D.h"

TEST(LinearLayerTest, ForwardPass) {
    nnm::Tensor4D input_tensor({
                                       {{{-0.766733f, -0.282161f, -0.768697f, -0.096156f, -0.128268f, 0.236904f,
                                          0.512347f},
                                         {0.061353f, 0.959301f, -0.164337f, 0.524297f, 0.799285f, -0.214173f,
                                          0.008204f},
                                         {0.457683f, 0.267843f, -0.031294f, -0.627326f, -0.185416f, -0.294899f,
                                          0.877054f},
                                         {0.162313f, -0.734360f, 0.186143f, 0.456118f, 0.350069f, -0.523494f,
                                          -0.710345f},
                                         {0.527861f, 0.840190f, 0.134171f, 0.981642f, -0.254548f, 0.491847f, 0.457749f},
                                         {-0.629972f, -0.654259f, 0.086984f, -0.621432f, -0.395667f, 0.463295f,
                                          -0.053500f}},
                                        {{-0.534607f, -0.499243f, 0.759338f, 0.063067f, -0.726757f, 0.440055f,
                                          0.746976f},
                                         {0.306095f, 0.295450f, -0.406594f, 0.068435f, -0.339947f, -0.250827f,
                                          -0.740342f},
                                         {-0.730286f, -0.067876f, 0.766455f, 0.041392f, -0.976632f, 0.331144f,
                                          0.265337f},
                                         {-0.015967f, -0.150896f, -0.237964f, -0.007849f, 0.336131f, 0.498830f,
                                          0.486832f},
                                         {0.826071f, -0.906265f, 0.995542f, -0.193859f, 0.419738f, -0.161364f,
                                          0.147551f},
                                         {-0.463937f, 0.799637f, -0.322525f, -0.447314f, -0.351903f, 0.132578f,
                                          -0.485165f}},
                                        {{0.392739f, -0.040489f, 0.019033f, 0.596817f, 0.056000f, -0.968945f,
                                          0.548904f},
                                         {0.998565f, -0.103700f, -0.458587f, 0.717230f, -0.070624f, -0.495174f,
                                          -0.851535f},
                                         {-0.526034f, -0.334167f, -0.825090f, -0.809140f, -0.651375f, -0.199416f,
                                          0.437921f},
                                         {-0.125070f, 0.646702f, 0.933709f, -0.424041f, 0.008524f, 0.287356f,
                                          0.735016f},
                                         {0.036863f, -0.411348f, 0.304255f, -0.314746f, 0.931622f, -0.691469f,
                                          -0.561478f},
                                         {0.325119f, -0.133885f, -0.458986f, 0.291829f, -0.813403f, -0.507098f,
                                          0.795703f}}}
                               });

// Weights
    nnm::Tensor4D weights({
                                  {{{0.047245f}, {0.065562f}, {0.048227f}, {-0.029931f}, {-0.012386f}, {0.035276f},
                                    {-0.020071f}, {0.058622f}, {-0.035407f}, {0.078628f}, {-0.034986f}, {-0.016069f},
                                    {-0.082987f}, {0.032516f}, {0.015885f}, {0.043463f}, {0.068727f}, {-0.020010f},
                                    {0.056036f}, {-0.061577f}, {0.017849f}, {0.054309f}, {-0.026255f}, {0.084086f},
                                    {0.074617f}, {-0.073398f}, {0.086096f}, {0.069048f}, {0.076892f}, {0.033807f},
                                    {-0.080029f}, {-0.032322f}, {0.025031f}, {0.082094f}, {-0.017156f}, {0.071526f},
                                    {0.075861f}, {0.004665f}, {-0.085677f}, {0.080501f}, {0.045935f}, {0.084064f},
                                    {-0.005149f}, {0.043864f}, {-0.071103f}, {-0.047197f}, {-0.034747f}, {0.082493f},
                                    {-0.061263f}, {0.062333f}, {0.008940f}, {-0.056433f}, {-0.025483f}, {-0.064693f},
                                    {-0.063738f}, {0.002934f}, {-0.071292f}, {0.038681f}, {0.085987f}, {0.064641f},
                                    {-0.057799f}, {0.012448f}, {-0.024335f}, {-0.008155f}, {-0.069048f}, {0.037995f},
                                    {-0.031884f}, {-0.019810f}, {0.037151f}, {0.088127f}, {0.063173f}, {-0.071005f},
                                    {0.010569f}, {-0.029852f}, {-0.067651f}, {-0.036561f}, {0.086639f}, {-0.037924f},
                                    {-0.052068f}, {0.036323f}, {0.020352f}, {-0.052905f}, {0.050151f}, {-0.080839f},
                                    {-0.019514f}, {0.016655f}, {-0.061251f}, {0.051286f}, {0.036407f}, {-0.040425f},
                                    {0.085875f}, {0.019397f}, {0.045318f}, {-0.033754f}, {-0.054408f}, {0.018831f},
                                    {0.015782f}, {0.000808f}, {-0.050780f}, {0.010571f}, {0.020569f}, {-0.067970f},
                                    {-0.062419f}, {-0.077454f}, {0.000469f}, {0.066714f}, {0.074520f}, {0.067776f},
                                    {-0.067731f}, {-0.000976f}, {-0.040718f}, {0.010271f}, {0.073514f}, {-0.031401f},
                                    {0.077437f}, {-0.061896f}, {-0.068381f}, {-0.077925f}, {-0.031408f}, {-0.000007f},
                                    {0.060329f}, {0.083828f}, {0.059686f}, {0.048254f}, {-0.085979f}, {0.048262f}},
                                   {{0.069488f}, {-0.057115f}, {0.029423f}, {-0.062339f}, {-0.081790f}, {-0.013499f},
                                    {-0.013097f}, {-0.022839f}, {0.076242f}, {-0.018193f}, {-0.032883f}, {-0.010931f},
                                    {-0.085571f}, {-0.066123f}, {-0.048678f}, {-0.012292f}, {0.060700f}, {0.052455f},
                                    {-0.018947f}, {0.079960f}, {0.059248f}, {-0.013537f}, {-0.018027f}, {0.018674f},
                                    {0.030576f}, {0.079912f}, {-0.070911f}, {0.052450f}, {0.060079f}, {-0.032112f},
                                    {-0.034098f}, {0.078892f}, {0.034198f}, {-0.068058f}, {0.050283f}, {-0.082308f},
                                    {-0.036764f}, {-0.006759f}, {0.017716f}, {-0.073802f}, {0.053427f}, {-0.055690f},
                                    {-0.069348f}, {-0.009013f}, {0.052899f}, {0.049792f}, {0.022169f}, {-0.012117f},
                                    {-0.077278f}, {-0.049895f}, {0.072662f}, {0.057891f}, {-0.059060f}, {-0.075435f},
                                    {0.089005f}, {0.047425f}, {-0.031942f}, {-0.018830f}, {-0.063069f}, {-0.013047f},
                                    {0.021004f}, {-0.061790f}, {0.059007f}, {0.016683f}, {-0.011821f}, {-0.085296f},
                                    {0.040704f}, {-0.008284f}, {-0.014515f}, {0.002518f}, {0.081760f}, {-0.038873f},
                                    {0.050820f}, {-0.039562f}, {-0.022716f}, {-0.074394f}, {0.052858f}, {0.002491f},
                                    {-0.056884f}, {-0.074991f}, {0.061092f}, {0.071957f}, {-0.038628f}, {0.051763f},
                                    {0.034540f}, {0.063780f}, {-0.073850f}, {-0.008883f}, {0.000276f}, {0.016607f},
                                    {-0.010904f}, {0.000649f}, {0.044151f}, {-0.082393f}, {-0.012324f}, {0.001151f},
                                    {-0.080397f}, {0.052390f}, {-0.006774f}, {0.021771f}, {0.014942f}, {-0.004300f},
                                    {-0.045042f}, {-0.063108f}, {0.060069f}, {-0.049191f}, {0.087326f}, {0.077032f},
                                    {0.032878f}, {0.060234f}, {-0.044020f}, {0.022126f}, {0.031257f}, {-0.078238f},
                                    {0.041221f}, {-0.060725f}, {-0.012413f}, {0.060225f}, {-0.035490f}, {-0.003772f},
                                    {0.004641f}, {0.056066f}, {-0.017228f}, {0.068112f}, {0.065844f}, {-0.016464f}},
                                   {{0.021981f}, {0.023929f}, {-0.088915f}, {-0.055384f}, {0.043978f}, {0.068684f},
                                    {0.007958f}, {-0.054624f}, {-0.026036f}, {0.080705f}, {-0.049832f}, {0.047821f},
                                    {0.050703f}, {0.029277f}, {-0.077851f}, {-0.052760f}, {-0.029557f}, {0.033126f},
                                    {-0.083383f}, {0.086685f}, {-0.067226f}, {0.009856f}, {0.055309f}, {-0.075220f},
                                    {-0.039968f}, {0.049505f}, {0.008945f}, {0.055246f}, {-0.011720f}, {0.041771f},
                                    {0.017590f}, {-0.003743f}, {-0.077971f}, {0.019501f}, {0.051943f}, {0.004075f},
                                    {-0.065818f}, {0.008786f}, {-0.009158f}, {0.085321f}, {0.068371f}, {0.023902f},
                                    {-0.045222f}, {0.053615f}, {0.081677f}, {0.039805f}, {-0.012788f}, {0.084744f},
                                    {-0.029136f}, {0.027785f}, {0.017402f}, {0.065270f}, {-0.007358f}, {0.051296f},
                                    {0.020858f}, {0.074851f}, {-0.008220f}, {-0.012532f}, {-0.015598f}, {0.021278f},
                                    {0.040735f}, {0.059367f}, {0.057127f}, {-0.046156f}, {0.062992f}, {-0.000245f},
                                    {0.038746f}, {-0.088424f}, {0.049492f}, {-0.076122f}, {0.083416f}, {-0.033410f},
                                    {0.080154f}, {0.088684f}, {0.060315f}, {-0.002858f}, {0.029984f}, {-0.009982f},
                                    {0.024978f}, {-0.073471f}, {-0.042417f}, {0.078116f}, {0.077136f}, {-0.087613f},
                                    {-0.039029f}, {-0.085420f}, {0.001221f}, {-0.057250f}, {-0.015967f}, {-0.077017f},
                                    {-0.017561f}, {-0.085827f}, {-0.000791f}, {0.052130f}, {0.027098f}, {-0.052003f},
                                    {-0.031771f}, {0.032280f}, {-0.050843f}, {0.002574f}, {0.016157f}, {0.014102f},
                                    {-0.001414f}, {-0.088878f}, {0.026159f}, {-0.022446f}, {-0.065334f}, {0.066524f},
                                    {-0.034515f}, {0.005613f}, {0.055015f}, {0.002652f}, {-0.002926f}, {0.026997f},
                                    {-0.027494f}, {-0.004116f}, {0.068233f}, {0.071653f}, {-0.034306f}, {0.061257f},
                                    {-0.044057f}, {0.084300f}, {-0.070235f}, {-0.082681f}, {0.044911f}, {-0.003387f}},
                                   {{-0.082447f}, {-0.037256f}, {0.011741f}, {-0.016343f}, {0.070056f}, {0.006066f},
                                    {-0.079842f}, {-0.072523f}, {0.001856f}, {0.052395f}, {-0.054051f}, {0.054587f},
                                    {-0.039088f}, {-0.044762f}, {-0.016166f}, {-0.025689f}, {-0.085671f}, {0.024218f},
                                    {0.021629f}, {-0.087919f}, {-0.055915f}, {0.052276f}, {0.081661f}, {-0.028574f},
                                    {0.043792f}, {-0.002342f}, {0.077869f}, {0.070039f}, {-0.057876f}, {0.037453f},
                                    {0.085707f}, {-0.074588f}, {0.011206f}, {-0.020920f}, {-0.011113f}, {-0.052869f},
                                    {0.067463f}, {0.083223f}, {-0.007242f}, {0.028780f}, {-0.023784f}, {-0.058716f},
                                    {-0.027990f}, {0.008339f}, {-0.059290f}, {0.035975f}, {-0.003470f}, {0.042995f},
                                    {0.005615f}, {0.028643f}, {-0.066439f}, {0.014771f}, {-0.083281f}, {0.017882f},
                                    {-0.085816f}, {-0.036550f}, {0.036448f}, {0.043302f}, {-0.081925f}, {0.012580f},
                                    {0.060376f}, {-0.006162f}, {-0.037177f}, {0.065603f}, {-0.017865f}, {-0.064447f},
                                    {-0.060176f}, {0.079496f}, {-0.000944f}, {-0.032964f}, {-0.001969f}, {0.006029f},
                                    {0.027652f}, {0.066941f}, {-0.075526f}, {0.063148f}, {-0.030658f}, {-0.022045f},
                                    {0.075205f}, {-0.015488f}, {0.040363f}, {0.015029f}, {-0.057260f}, {0.067375f},
                                    {-0.036664f}, {-0.038794f}, {0.039551f}, {-0.051316f}, {0.024179f}, {-0.079646f},
                                    {-0.036717f}, {-0.027860f}, {0.077595f}, {-0.057016f}, {-0.042947f}, {-0.015824f},
                                    {-0.024826f}, {0.041477f}, {0.075114f}, {-0.008462f}, {0.070744f}, {0.037086f},
                                    {0.074088f}, {-0.018308f}, {0.088530f}, {-0.000492f}, {-0.062008f}, {0.065261f},
                                    {-0.025099f}, {-0.072994f}, {-0.034453f}, {-0.088342f}, {0.067402f}, {-0.071580f},
                                    {-0.069021f}, {-0.087979f}, {-0.088123f}, {-0.027338f}, {0.002745f}, {-0.002553f},
                                    {0.005993f}, {-0.053498f}, {0.047657f}, {-0.000259f}, {-0.012833f}, {-0.058541f}},
                                   {{0.013432f}, {0.075298f}, {-0.063722f}, {0.018739f}, {-0.074921f}, {0.010376f},
                                    {-0.058558f}, {0.046197f}, {0.005127f}, {0.008459f}, {-0.021328f}, {0.074533f},
                                    {-0.010169f}, {-0.033711f}, {0.087228f}, {-0.085303f}, {-0.036993f}, {0.088515f},
                                    {-0.007485f}, {0.027686f}, {-0.028933f}, {-0.037930f}, {0.086614f}, {0.026368f},
                                    {0.075935f}, {0.008858f}, {-0.057361f}, {-0.056693f}, {0.009561f}, {-0.051024f},
                                    {-0.052647f}, {-0.007359f}, {-0.034714f}, {0.059758f}, {-0.019652f}, {0.028971f},
                                    {0.084150f}, {0.007902f}, {0.027057f}, {-0.043376f}, {0.043538f}, {-0.086539f},
                                    {-0.019691f}, {0.008754f}, {0.025830f}, {-0.038060f}, {0.083784f}, {0.009161f},
                                    {-0.048970f}, {0.017277f}, {-0.083750f}, {-0.065239f}, {0.027294f}, {0.077961f},
                                    {0.058813f}, {0.046390f}, {0.044023f}, {-0.085023f}, {0.026984f}, {-0.007956f},
                                    {0.080342f}, {-0.002952f}, {-0.063179f}, {-0.050419f}, {-0.014334f}, {0.037195f},
                                    {0.080774f}, {-0.055976f}, {-0.072993f}, {-0.024037f}, {0.042566f}, {0.050181f},
                                    {-0.054802f}, {-0.002025f}, {0.001788f}, {-0.016121f}, {0.080379f}, {-0.075719f},
                                    {-0.018931f}, {0.055419f}, {-0.040756f}, {-0.088717f}, {0.025410f}, {0.084152f},
                                    {0.084466f}, {0.050661f}, {-0.073532f}, {-0.054263f}, {-0.087061f}, {-0.004203f},
                                    {0.049686f}, {0.026562f}, {0.088097f}, {0.052740f}, {0.064480f}, {-0.008037f},
                                    {0.067253f}, {-0.076187f}, {-0.085188f}, {-0.072122f}, {0.057173f}, {0.017022f},
                                    {-0.009042f}, {0.029372f}, {-0.016220f}, {0.005865f}, {-0.022620f}, {-0.072451f},
                                    {0.040443f}, {0.025117f}, {0.068336f}, {-0.026027f}, {0.026160f}, {0.011664f},
                                    {-0.045668f}, {-0.055620f}, {0.059373f}, {-0.069961f}, {0.009022f}, {-0.083892f},
                                    {-0.053226f}, {-0.058546f}, {0.050970f}, {0.025280f}, {-0.054455f}, {-0.050599f}},
                                   {{-0.077432f}, {-0.047143f}, {-0.034183f}, {-0.017001f}, {-0.079374f}, {-0.003593f},
                                    {0.012566f}, {-0.084029f}, {-0.008858f}, {-0.061692f}, {0.088613f}, {0.012799f},
                                    {-0.055983f}, {-0.053041f}, {-0.069064f}, {-0.073813f}, {0.084477f}, {0.061632f},
                                    {-0.056206f}, {-0.025146f}, {0.062317f}, {0.056915f}, {-0.030969f}, {0.049767f},
                                    {0.011581f}, {-0.026102f}, {0.067595f}, {-0.063625f}, {0.001041f}, {0.042618f},
                                    {0.015331f}, {-0.027064f}, {-0.008700f}, {-0.046111f}, {0.023456f}, {-0.075062f},
                                    {0.001629f}, {-0.067994f}, {-0.000643f}, {-0.047065f}, {-0.088985f}, {0.006044f},
                                    {0.052167f}, {-0.082326f}, {-0.085857f}, {0.010257f}, {-0.004756f}, {-0.005561f},
                                    {-0.002715f}, {0.039276f}, {0.074381f}, {-0.004372f}, {0.061149f}, {0.049672f},
                                    {0.048634f}, {0.034422f}, {0.065367f}, {-0.032371f}, {-0.020016f}, {-0.007662f},
                                    {0.081772f}, {-0.081482f}, {0.053580f}, {0.061674f}, {-0.020565f}, {-0.066913f},
                                    {0.060565f}, {-0.042377f}, {0.087951f}, {0.058225f}, {-0.054095f}, {0.023077f},
                                    {-0.047458f}, {-0.025682f}, {-0.055697f}, {0.007631f}, {0.035929f}, {-0.033826f},
                                    {-0.047712f}, {0.072333f}, {0.072036f}, {0.036170f}, {-0.004380f}, {0.027174f},
                                    {-0.046258f}, {-0.021027f}, {0.056632f}, {-0.040876f}, {0.004727f}, {-0.061817f},
                                    {0.036710f}, {-0.075886f}, {-0.025542f}, {-0.065836f}, {0.003412f}, {0.019137f},
                                    {0.072076f}, {0.002504f}, {-0.050332f}, {-0.048527f}, {-0.069969f}, {-0.074145f},
                                    {-0.049175f}, {0.055409f}, {-0.016121f}, {-0.013599f}, {0.077863f}, {-0.076084f},
                                    {0.078428f}, {0.043348f}, {0.088019f}, {0.051139f}, {0.001880f}, {-0.057240f},
                                    {-0.024254f}, {0.041568f}, {0.061804f}, {0.056198f}, {-0.011220f}, {0.058745f},
                                    {0.053594f}, {-0.068488f}, {-0.030064f}, {0.053201f}, {-0.081043f}, {-0.000614f}},
                                   {{0.079253f}, {0.004136f}, {-0.066456f}, {-0.010342f}, {-0.000479f}, {0.015904f},
                                    {0.035158f}, {-0.073493f}, {-0.083766f}, {-0.088043f}, {-0.038799f}, {0.054932f},
                                    {0.023119f}, {-0.061789f}, {-0.021963f}, {-0.027463f}, {-0.040108f}, {0.004111f},
                                    {-0.079954f}, {-0.081668f}, {-0.044007f}, {-0.021446f}, {-0.042198f}, {-0.076019f},
                                    {-0.032039f}, {0.047031f}, {-0.023683f}, {0.019857f}, {0.061772f}, {-0.026108f},
                                    {-0.007723f}, {0.046289f}, {-0.050542f}, {0.036051f}, {0.069558f}, {0.084964f},
                                    {0.052821f}, {-0.061522f}, {-0.029460f}, {-0.041780f}, {0.082939f}, {-0.031306f},
                                    {0.070521f}, {0.083798f}, {-0.018422f}, {-0.062641f}, {-0.044580f}, {-0.032273f},
                                    {0.055713f}, {-0.075435f}, {0.027468f}, {0.063622f}, {-0.079616f}, {0.047602f},
                                    {0.011162f}, {-0.025183f}, {-0.031870f}, {0.045069f}, {0.048372f}, {0.080901f},
                                    {0.037741f}, {0.064758f}, {-0.087603f}, {0.022040f}, {0.000647f}, {-0.050365f},
                                    {0.055574f}, {0.026488f}, {-0.058152f}, {0.034414f}, {0.021922f}, {0.026830f},
                                    {-0.006264f}, {-0.040318f}, {-0.049515f}, {-0.023630f}, {-0.074368f}, {0.082019f},
                                    {0.079910f}, {-0.066674f}, {-0.028589f}, {-0.059085f}, {0.033219f}, {0.043554f},
                                    {-0.032515f}, {0.001387f}, {-0.080777f}, {0.007844f}, {0.064996f}, {-0.010549f},
                                    {0.032693f}, {-0.011048f}, {0.082722f}, {0.060059f}, {-0.025755f}, {-0.077828f},
                                    {-0.070945f}, {-0.087689f}, {-0.075101f}, {0.044476f}, {0.054188f}, {-0.034031f},
                                    {0.068093f}, {-0.038143f}, {0.087092f}, {0.016104f}, {-0.012322f}, {-0.009764f},
                                    {0.025650f}, {0.016878f}, {0.036444f}, {0.066065f}, {0.032707f}, {0.080844f},
                                    {0.041211f}, {-0.058667f}, {-0.087331f}, {-0.000197f}, {-0.037682f}, {0.029940f},
                                    {0.059166f}, {-0.003850f}, {0.075385f}, {-0.040820f}, {-0.038729f}, {-0.037058f}},
                                   {{0.086914f}, {-0.004664f}, {0.054126f}, {0.083011f}, {0.035166f}, {0.030848f},
                                    {0.050969f}, {0.052383f}, {-0.014942f}, {-0.025580f}, {-0.023916f}, {-0.046406f},
                                    {0.075717f}, {-0.066200f}, {-0.039082f}, {-0.066063f}, {-0.084049f}, {0.043957f},
                                    {0.075484f}, {0.053261f}, {-0.036483f}, {-0.069521f}, {-0.017805f}, {0.029079f},
                                    {-0.082468f}, {0.016580f}, {-0.077095f}, {0.073165f}, {-0.072949f}, {-0.032813f},
                                    {0.079375f}, {-0.081360f}, {0.057450f}, {-0.047381f}, {0.078622f}, {0.074271f},
                                    {0.062045f}, {0.028843f}, {0.067619f}, {0.055197f}, {-0.008626f}, {0.015096f},
                                    {0.040779f}, {-0.058024f}, {-0.027867f}, {-0.021187f}, {-0.050667f}, {-0.088168f},
                                    {-0.039633f}, {0.007697f}, {0.067242f}, {-0.055157f}, {0.058051f}, {-0.013518f},
                                    {-0.053496f}, {-0.031341f}, {0.036764f}, {0.031041f}, {0.004917f}, {-0.049770f},
                                    {-0.078146f}, {0.087968f}, {0.009160f}, {0.067188f}, {0.051753f}, {-0.088032f},
                                    {-0.046854f}, {0.003052f}, {-0.056674f}, {-0.017927f}, {0.053485f}, {0.053392f},
                                    {0.027582f}, {0.035348f}, {-0.066032f}, {0.065418f}, {-0.014419f}, {0.085402f},
                                    {0.061503f}, {-0.065238f}, {0.052392f}, {0.049530f}, {-0.057125f}, {-0.060476f},
                                    {-0.034907f}, {0.069664f}, {-0.080974f}, {-0.021725f}, {-0.070345f}, {-0.085625f},
                                    {0.027922f}, {-0.068736f}, {-0.067854f}, {0.076047f}, {-0.073279f}, {-0.034938f},
                                    {-0.030428f}, {-0.084604f}, {-0.037795f}, {-0.037538f}, {-0.033983f}, {0.017735f},
                                    {-0.052597f}, {0.081099f}, {0.070973f}, {-0.028500f}, {-0.013776f}, {0.065787f},
                                    {-0.033050f}, {-0.058630f}, {-0.006561f}, {0.017876f}, {0.054526f}, {0.062087f},
                                    {0.010262f}, {0.082507f}, {0.028962f}, {-0.026745f}, {-0.067408f}, {0.023361f},
                                    {-0.085265f}, {0.037525f}, {0.059016f}, {-0.063066f}, {0.009472f}, {0.017735f}},
                                   {{0.021459f}, {0.068599f}, {-0.047194f}, {0.073419f}, {-0.033759f}, {0.011602f},
                                    {0.013602f}, {0.001495f}, {0.035314f}, {0.072348f}, {0.022981f}, {0.053829f},
                                    {-0.050948f}, {0.012988f}, {0.047343f}, {-0.008304f}, {0.029051f}, {-0.062311f},
                                    {0.038591f}, {0.013535f}, {-0.053283f}, {0.028734f}, {-0.038118f}, {0.010439f},
                                    {-0.014469f}, {-0.030417f}, {0.029918f}, {0.067177f}, {-0.050505f}, {-0.072876f},
                                    {-0.016079f}, {0.084891f}, {-0.070021f}, {0.063025f}, {0.017426f}, {0.055424f},
                                    {0.051744f}, {0.004024f}, {-0.045482f}, {-0.003377f}, {0.005026f}, {-0.037012f},
                                    {0.019280f}, {-0.074258f}, {0.067246f}, {0.071736f}, {-0.009206f}, {0.002831f},
                                    {-0.073679f}, {-0.077705f}, {-0.009819f}, {-0.011367f}, {-0.068069f}, {0.023876f},
                                    {-0.011294f}, {-0.044922f}, {-0.007047f}, {-0.025589f}, {0.060890f}, {-0.086992f},
                                    {-0.008100f}, {-0.076913f}, {0.004612f}, {-0.036865f}, {0.049288f}, {-0.067044f},
                                    {0.052377f}, {0.014028f}, {0.064175f}, {-0.086467f}, {-0.056661f}, {0.037996f},
                                    {0.069387f}, {0.037580f}, {-0.075419f}, {0.051035f}, {0.088657f}, {-0.073060f},
                                    {-0.041438f}, {-0.049327f}, {-0.060023f}, {-0.054431f}, {-0.032810f}, {-0.074148f},
                                    {0.059244f}, {-0.025650f}, {-0.039650f}, {0.069671f}, {0.058475f}, {0.033516f},
                                    {0.001792f}, {0.030773f}, {0.044416f}, {0.007067f}, {-0.026048f}, {-0.081559f},
                                    {0.066152f}, {0.084970f}, {-0.035695f}, {-0.040415f}, {-0.055313f}, {-0.041350f},
                                    {-0.013231f}, {-0.058482f}, {-0.065228f}, {0.065292f}, {0.087009f}, {-0.045898f},
                                    {0.025035f}, {-0.006191f}, {0.023367f}, {0.016292f}, {0.071124f}, {-0.077640f},
                                    {0.004174f}, {0.019438f}, {0.026078f}, {0.066685f}, {-0.023243f}, {0.011758f},
                                    {0.038525f}, {0.073977f}, {-0.067361f}, {0.079740f}, {0.058799f}, {-0.073992f}},
                                   {{-0.012674f}, {-0.013966f}, {-0.072176f}, {-0.035355f}, {0.042599f}, {0.005498f},
                                    {-0.071815f}, {0.082331f}, {-0.088320f}, {0.044737f}, {-0.026243f}, {-0.030379f},
                                    {-0.063869f}, {-0.067589f}, {0.051558f}, {0.043643f}, {0.082191f}, {-0.025000f},
                                    {0.061653f}, {-0.001569f}, {0.064829f}, {-0.088077f}, {-0.068251f}, {0.040305f},
                                    {0.045568f}, {-0.026977f}, {-0.026207f}, {-0.068387f}, {-0.002607f}, {0.084536f},
                                    {0.058343f}, {0.031914f}, {-0.073375f}, {0.016801f}, {0.050886f}, {0.001105f},
                                    {0.036906f}, {0.001158f}, {0.001835f}, {0.087468f}, {-0.044797f}, {-0.029368f},
                                    {0.016472f}, {0.043268f}, {-0.028791f}, {-0.022958f}, {0.026427f}, {0.046501f},
                                    {-0.088314f}, {-0.083623f}, {-0.056130f}, {-0.012112f}, {0.025909f}, {-0.084028f},
                                    {-0.071488f}, {0.053592f}, {-0.036155f}, {-0.072428f}, {-0.061962f}, {0.012294f},
                                    {0.078815f}, {-0.067489f}, {-0.076122f}, {-0.007959f}, {0.057140f}, {-0.034766f},
                                    {-0.066958f}, {0.088921f}, {-0.078429f}, {0.052380f}, {-0.022822f}, {0.043763f},
                                    {-0.045145f}, {-0.081867f}, {-0.042236f}, {-0.057051f}, {0.008047f}, {0.057552f},
                                    {0.068799f}, {-0.034647f}, {0.072745f}, {-0.058882f}, {-0.087947f}, {-0.078237f},
                                    {-0.054469f}, {0.049154f}, {0.034335f}, {-0.072294f}, {-0.009259f}, {-0.006651f},
                                    {0.059426f}, {0.019610f}, {0.033541f}, {0.035396f}, {0.022421f}, {-0.049908f},
                                    {0.037196f}, {0.009280f}, {0.026725f}, {0.050542f}, {0.065738f}, {0.042332f},
                                    {-0.076317f}, {-0.031362f}, {-0.049931f}, {-0.068248f}, {-0.070380f}, {-0.076861f},
                                    {0.018964f}, {0.058589f}, {0.058984f}, {0.012592f}, {0.016693f}, {0.052289f},
                                    {-0.052262f}, {0.084187f}, {0.075227f}, {0.050967f}, {0.019759f}, {-0.035286f},
                                    {0.058288f}, {-0.035125f}, {0.005799f}, {-0.062632f}, {0.014768f}, {0.067398f}}}
                          });

// Bias
    nnm::Tensor4D bias({
                               {{{0.064497f}}, {{-0.019735f}}, {{-0.058033f}}, {{-0.034855f}}, {{0.062204f}},
                                {{-0.067866f}}, {{-0.081291f}}, {{-0.014824f}}, {{-0.081683f}}, {{0.066617f}}}
                       });

// Expected output tensor
    nnm::Tensor4D expected_output({
                                          {{{0.731957f}}, {{0.262660f}}, {{0.103667f}}, {{-0.683848f}}, {{-0.243974f}},
                                           {{-0.011355f}}, {{0.047253f}}, {{-0.239260f}}, {{-0.024706f}},
                                           {{-0.183299f}}}
                                  });

    // Create LinearLayer
    size_t in_features = input_tensor.getChannels() * input_tensor.getHeight() * input_tensor.getWidth();
    size_t out_features = expected_output.getChannels();
    nnm::LinearLayer linear_layer(in_features, out_features);

    // Set weights and bias
    linear_layer.set_weights(weights);
    linear_layer.set_bias(bias);

    // Perform forward pass
    nnm::Tensor4D output = linear_layer.forward(input_tensor);

    // Compare output with expected output
    ASSERT_EQ(output.getBatchSize(), expected_output.getBatchSize());
    ASSERT_EQ(output.getChannels(), expected_output.getChannels());
    ASSERT_EQ(output.getHeight(), expected_output.getHeight());
    ASSERT_EQ(output.getWidth(), expected_output.getWidth());

    for (size_t n = 0; n < output.getBatchSize(); ++n) {
        for (size_t c = 0; c < output.getChannels(); ++c) {
            for (size_t h = 0; h < output.getHeight(); ++h) {
                for (size_t w = 0; w < output.getWidth(); ++w) {
                    EXPECT_NEAR(output(n, c, h, w), expected_output(n, c, h, w), 1e-4)
                                        << "Mismatch at position (" << n << ", " << c << ", " << h << ", " << w << ")";
                }
            }
        }
    }
}

