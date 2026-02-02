VLMaps introduces a novel spatial map representation that integrates pretrained visual-language features directly into a 3D reconstruction of the physical world. By fusing dense embeddings from models like LSeg with geometric data, this system allows robots to localize open-vocabulary landmarks and complex spatial references without requiring manual labels or additional training. When paired with Large Language Models (LLMs), the framework translates natural language commands into executable Python code for precise navigation.

<video width="900" controls>
    <source src="https://logus2k.com/docbro/categories/robotics/videos/VLMaps__Robot_Understanding.mp4" type="video/mp4">
</video>

A significant advantage of this approach is its ability to generate customized obstacle maps on-the-fly for different robot embodiments, such as drones or ground vehicles. Experimental results in both simulated and real-world environments demonstrate that VLMaps outperforms existing methods in handling long-horizon, language-guided tasks. Ultimately, this research bridges the gap between high-level linguistic reasoning and low-level spatial precision in robotics.
