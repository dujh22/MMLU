该文件包含多任务测试的 dev、val 和 test 数据。

dev 数据集用于少量学习，为模型打基础，而测试集则是评估问题的来源。

auxiliary_training 数据可用于微调，这对没有少量学习能力的模型很重要。

这些辅助训练数据来自其他 NLP 多选数据集，如 MCTest（Richardson 等人，2013 年）、RACE（Lai 等人，2017 年）、ARC（Clark 等人，2018 年，2016 年）和 OBQA（Mihaylov 等人，2018 年）。

除非另有说明，这些问题均以截至 2020 年 1 月 1 日的人类知识为参考。在遥远的将来，在提示语中添加 "该问题是为 2020 年的受众编写的 "可能会有所帮助。

--

如果您发现这对您的研究有用，请考虑引用该测试及其所使用的 ETHICS 数据集：

@article{hendryckstest2021,
  title={Measuring Massive Multitask Language Understanding},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}

@article{hendrycks2021ethics,
  title={Aligning AI With Shared Human Values},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
