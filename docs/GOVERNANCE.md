# GPJax Governance Document

## The Project

GPJax is an open-source library that supports Gaussian process modelling in the JAX
scientific computation ecosystem. The abstractions provided in GPJax are designed to
mimic the underlying maths through, making the library easy to use for both researchers
and practitioners alike.

GPJax was created by [Thomas Pinder](https://github.com/thomaspinder) as a
[Single-Maintainer Houseplant
project](https://github.com/OpenTechStrategies/open-source-archetypes/blob/main/arch-houseplant.ltx)
following the BDFL model of governance. We have since moved to the governance model of
[Specialty
Library](https://github.com/OpenTechStrategies/open-source-archetypes/blob/main/arch-specialty-library.ltx)
and benefited from a community of
[contributors](https://github.com/thomaspinder/GPJax/graphs/contributors). This
document outlines the governance structure for the current status.

## Roles
* Contributors: Anyone who contributes to GPJAx is considered a contributor. This
  includes submitting code, filing issues, reviewing pull requests, and participating in
  discussions. They are listed under:
   * [https://github.com/thomaspinder/GPJax/graphs/contributors](https://github.com/thomaspinder/GPJax/graphs/contributors)
* Core contributors: Core contributors are contributors who have made significant
  contributions to the GPJax project, for example large modules or functionality.
* GPJax gardeners: Gardeners are core contributor(s) who are responsible for maintaining
  the project and making decisions about its future direction. A GPJax gardener(s) has the
  ability to merge pull requests into the GPJax repository. GPJax gardeners also take on
  administrative tasks such as website maintenance.
   * Currently  [thomaspinder@](https://github.com/thomaspinder) is the gardener of GPJax.

## Responsibility
We cannot hold anyone responsible really since we are all doing free work here, but some
general expectations are:
* Contributors are responsible for following the project's code of conduct and
  contributing to the project in a positive and constructive manner. Contributors are
  also responsible for testing their code and ensuring that it meets the project's
  standards.
* Core contributors are expected to review pull requests and provide feedback to
  contributors. They also make decisions about the architecture and implementation of
  the module/functionality they contributed to. Also the “if you broke something please
  fix it” applies.
* Maintainers are responsible for monitoring the benchmark, the documentation and the
  website are up to date and built passed, update dependency and apply best practice

In addition to these specific responsibilities, all contributors are encouraged to
participate in discussions about the project and to help out in any way they can.

## Decision-making
Decisions about the GPJax project are made by consensus among the GPJax gardeners. This
means that all GPJax gardeners must agree on a decision before it can be implemented. If
a consensus cannot be reached, we will flip a (virtual) coin.

## Communication
Communication about the GPJax project takes place in the following channels:

GitHub issues: Issues are used to track bugs, feature requests, and other tasks. GitHub
discussion: Discussions are used to answer user questions, scope for features, and
discuss solutions to bugs. GitHub pull requests: Pull requests propose changes to the
GPJax codebase. Slack: The GPJax Slack Channel is used for internal communication about
the project for core contributors.

## Contributing
Anyone is welcome to contribute to the GPJax project. Contributions can be made in the
form of code, documentation, or other forms of support. To learn more about how to
contribute, please see the [contributing
guide](https://github.com/thomaspinder/GPJax/blob/main/static/CONTRIBUTING.md).


## Code of conduct
All contributors to the GPJax project are expected to follow the project's [code of
conduct](https://github.com/thomaspinder/GPJax/blob/main/.github/CODE_OF_CONDUCT.md).
The code of conduct outlines the expected behavior of contributors and helps to ensure a
welcoming and productive environment for all.

Any breaches of the code of conduct should be reported using our [contact
form](https://jaxgaussianprocesses.com/contact/).


## Governance changes
This governance document is subject to change. Changes to the governance document must
be approved by consensus among the core contributors.


## Contact
If you have any questions about the GPJax project, please feel free to contact the
maintainers or reach out over
[Slack](https://join.slack.com/t/gpjax/shared_invite/zt-1da57pmjn-rdBCVg9kApirEEn2E5Q2Zw).

-----

This file was adapted from
[BlackJAX](https://github.com/blackjax-devs/blackjax/blob/main/GOVERNANCE.md).
