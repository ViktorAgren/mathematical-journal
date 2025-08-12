import React from "react";
import { CodeWindow } from "../../components/CodeWindow";
import { LatexBlock } from "../../components/LatexBlock";
import { Theorem } from "../../components/Theorem";
import { Note } from "../../components/Note";
import { InlineMath } from "../../components/InlineMath";

export const SoccerPositionModelsContent = () => {
  return (
    <>
      <section className="paper-section">
        <h2 className="section-title">
          The £100 Million Transfer Mistake
        </h2>

        <p>
          Picture this: It's summer 2019, and Barcelona just paid €120 million for Antoine Griezmann. On paper, it made perfect sense – he scored 21 goals and provided 10 assists the previous season. His "key passes per game" were elite. His "progressive runs" looked impressive. Every traditional metric screamed "world-class forward." Fast forward two years, and Barcelona was desperately trying to offload him to Atlético Madrid.
        </p>

        <p>
          What went wrong? The same thing that goes wrong with most big-money transfers: everyone was looking at the wrong numbers. Goals and assists tell you what happened, not why it happened. They're like looking at a company's quarterly earnings without understanding the underlying business model. You might get lucky and buy a winner, or you might spend €120 million on someone whose playing style fundamentally doesn't match what your team actually needs.
        </p>

        <p>
          The revolution in soccer analytics isn't about counting more things – it's about understanding <em>value</em>. Every time a player touches the ball, they're making a decision that either increases or decreases their team's chances of scoring the next goal. That simple insight has spawned a mathematical framework that's transforming how the smartest clubs evaluate, buy, and deploy talent.
        </p>

        <h2 className="section-title">
          Why Your Fantasy Football Team Keeps Losing
        </h2>

        <p>
          If you've ever played fantasy football, you know the frustration. You pick the striker who scored 25 goals last season, and he proceeds to disappear for the next three months. Meanwhile, your friend who selected some unknown midfielder from Brighton is collecting points every week with a player who rarely scores but somehow always seems to be "involved" in good things happening.
        </p>

        <p>
          This isn't bad luck – it's the difference between measuring <em>output</em> and measuring <em>process</em>. Your 25-goal striker might have been the lucky beneficiary of a system that created chances for him. The Brighton midfielder might be the one actually creating value, game after game, in ways that traditional statistics completely miss.
        </p>

        <p>
          Here's the key insight: football is a probabilistic game played on a continuous surface. Every moment, every position, every touch of the ball represents a different probability of eventually scoring or conceding. When Kevin De Bruyne receives the ball 40 yards from goal, that situation has a certain expected value. When he threads a pass between three defenders to put Erling Haaland through on goal, that expected value changes dramatically. The difference between those two values? That's De Bruyne's contribution to that moment.
        </p>

        <h2 className="section-title">
          The Math Behind the Magic
        </h2>

        <p>
          The mathematical breakthrough that's revolutionizing soccer analytics has a deceptively simple name: Expected Possession Value, or EPV. Think of it as the stock price of every moment in a football match. Just like a stock price reflects the market's collective wisdom about a company's future prospects, EPV reflects the mathematical expectation of what will happen next in the game.
        </p>

        <p>
          Formally, we can write this as:
        </p>

        <LatexBlock equation="EPV_t = \mathbb{E}[G | T_t]" />

        <p>
          where <InlineMath tex="G" /> is the possession outcome (1 if your team scores next, -1 if the opponent scores, 0 if the possession fizzles out) and <InlineMath tex="T_t" /> represents everything happening on the pitch at time <InlineMath tex="t" /> – player positions, ball location, game phase, you name it. It's like having a crystal ball that can see all possible futures and telling you, on average, what's most likely to happen.
        </p>

        <p>
          But here's where it gets interesting for player evaluation. EPV doesn't just tell us what might happen – it tells us how much each player's actions change what might happen. When Luka Modrić receives the ball deep in midfield and plays a 40-yard diagonal pass that bypasses three opponents, he's not just completing a pass – he's fundamentally altering the probability landscape of the game.
        </p>

        <p>
          This is where the real innovation comes in. Instead of just measuring the final EPV, we can decompose it into all the little decisions and actions that led there:
        </p>

        <LatexBlock equation="EPV_t = \sum_{a \in A} P(a | T_t) \cdot \mathbb{E}[G | a, T_t]" />

        <p>
          Think of this equation as a map of all possible futures. Each action <InlineMath tex="a" /> (pass, dribble, shot, tackle) has some probability <InlineMath tex="P(a | T_t)" /> of happening in the current situation, and each action leads to some expected outcome <InlineMath tex="\mathbb{E}[G | a, T_t]" />. The player's job is to choose the action that maximizes this value, and our job is to measure how well they do it.
        </p>

        <p>
          Now, there are different ways to slice and dice this value creation, and each approach reveals different aspects of the game. The first breakthrough was called Expected Threat, or xT, and it's beautifully simple in concept. Imagine overlaying a heat map on the football pitch where each zone has a number representing how dangerous it is to have the ball there. The penalty box? Very dangerous. The center circle? Not so much. Your own penalty area? Dangerous, but for the wrong team.
        </p>

        <p>
          Expected Threat captures this by calculating:
        </p>

        <LatexBlock equation="xT_{action} = xT_{end} - xT_{start}" />

        <p>
          Every time a player moves the ball from one zone to another, we can quantify exactly how much more (or less) dangerous the situation became. When Trent Alexander-Arnold whips in one of his trademark crosses from the right flank, he's not just "creating a chance" – he's moving the ball from a zone worth, say, 0.05 goal probability to a zone worth 0.25. That difference? That's his contribution to Liverpool's attack.
        </p>

        <p>
          But xT was just the beginning. The really sophisticated stuff came when analysts realized they needed to account for both sides of the ball. Enter On-Ball Value, or OBV, which recognizes that every action has both an attacking and defending component:
        </p>

        <LatexBlock equation="OBV_{action} = OBV_{for} + OBV_{against}" />

        <p>
          This might sound abstract, but think about it this way: when N'Golo Kanté makes one of his trademark interceptions in midfield, he's not just stopping an attack – he's simultaneously destroying the opponent's possession value and creating value for his own team. The equation above captures both sides of that coin. It's the difference between measuring what a player does <em>to</em> the game versus what they do <em>for</em> their team.
        </p>

        <Theorem title="The Conservation Law of Football">
          <p>
            Here's a beautiful mathematical property that makes all this work: possession value is conserved. For any sequence of play, the sum of all individual contributions equals the total change in the possession's value:
          </p>
          <LatexBlock equation="\sum_{i=1}^{n} \Delta EPV_i = EPV_{end} - EPV_{start}" />
          <p>
            This isn't just a neat mathematical trick – it's what allows us to fairly distribute credit and blame across all 22 players on the pitch. Every gram of value created or destroyed can be traced back to someone's decision.
          </p>
        </Theorem>

        <h2 className="section-title">
          Why Virgil van Dijk Is Worth More Than Erling Haaland (Sometimes)
        </h2>
        
        <p>
          Here's where things get really interesting, and where most traditional analysis falls apart. Not all players contribute value in the same way, and not all value contributions are created equal. Asking whether Haaland or van Dijk is "better" is like asking whether a Ferrari engine or a Formula 1 chassis is more important – it depends entirely on what you're trying to achieve and how you measure success.
        </p>

        <p>
          The mathematical breakthrough comes from recognizing that each position has its own unique value signature. We can model this with position-specific weightings:
        </p>

        <LatexBlock equation="V_{pos}(i) = \sum_{j \in J_{pos}} w_{pos,j} \cdot EPV_j(i)" />

        <p>
          Think of <InlineMath tex="w_{pos,j}" /> as position-specific multipliers that recognize what matters most for each role. For a striker, finishing might get a weight of 0.4, while for a center-back, it might be 0.05. It's not that center-backs can't score (hello, Sergio Ramos), but that's not primarily what you're paying them to do.
        </p>

        <p>
          Let's break this down with some real examples. For forwards like Haaland, we weight heavily toward goal-scoring and advanced positioning:
        </p>

        <LatexBlock equation="V_{FW}(i) = \omega_1 \cdot xG_i + \omega_2 \cdot xT_{progressive,i} + \omega_3 \cdot OBV_{final\_third,i}" />

        <p>
          This captures what forwards actually do: finish chances, get into dangerous positions, and make things happen in the final third. When Haaland makes one of his trademark runs into the box, he's not just running – he's moving from a zone where his team has a 15% chance of scoring to a zone where they have a 35% chance. That 20-point increase? That's measurable value creation.
        </p>

        <p>
          Midfielders like De Bruyne require a more balanced approach:
        </p>

        <LatexBlock equation="V_{MF}(i) = \omega_1 \cdot xT_{pass,i} + \omega_2 \cdot OBV_{build\_up,i} + \omega_3 \cdot EPV_{defensive,i}" />

        <p>
          Midfielders are the Swiss Army knives of football – they need to create, destroy, build, and transition. The model reflects this by weighing their passing creativity, possession building, and defensive contributions roughly equally. It's why Kevin De Bruyne can have an "off day" goal-wise but still dominate the EPV charts through his passing and build-up play.
        </p>

        <p>
          And then there are defenders like van Dijk, whose value often appears in what <em>doesn't</em> happen:
        </p>

        <LatexBlock equation="V_{DF}(i) = \omega_1 \cdot (-EPV_{prevented,i}) + \omega_2 \cdot OBV_{against,i} + \omega_3 \cdot xT_{distribution,i}" />

        <p>
          Notice that negative sign in front of <InlineMath tex="EPV_{prevented}" />. Van Dijk's value often comes from preventing opponent chances rather than creating his own team's chances. When he makes one of his perfectly-timed interceptions, he's not just winning the ball – he's destroying what might have been a 0.3 EPV situation for the opponent and turning it into a 0.1 EPV situation for his team. That's a 0.4 EPV swing that never shows up in traditional statistics.
        </p>

        <h2 className="section-title">
          Building Your Own Moneyball
        </h2>

        <p>
          All of this mathematical sophistication ultimately serves one purpose: making better decisions about which players to buy, sell, and deploy. The possession value framework gives us a foundation for predicting market values that goes far beyond traditional statistics:
        </p>

        <LatexBlock equation="M = f(\text{EPV}, \text{Age}, \text{Contract}, \text{Market}) + \epsilon" />

        <p>
          Think of this as the ultimate transfer algorithm. Performance is captured through EPV metrics rather than goals and assists. Age effects are modeled through the well-known career curves that peak around 27 for most outfield players:
        </p>

        <LatexBlock equation="P(t) = \beta_0 + \beta_1 t + \beta_2 t^2 + \epsilon" />

        <p>
          The beauty of this approach is that it reveals inefficiencies in the transfer market. When Brighton consistently finds players who outperform their transfer fees, they're not getting lucky – they're systematically identifying players whose possession value metrics exceed their market price. They're buying EPV at a discount.
        </p>

        <Note>
          <p>
            <strong>The Brighton Model:</strong> Brighton's recruitment success isn't magic – it's mathematics. They systematically target players whose possession value metrics exceed their market price, particularly from less-watched leagues. Their model has turned a small budget into consistent Premier League survival and significant transfer profits.
          </p>
        </Note>

        <p>
          The revolution in soccer analytics isn't just about having better numbers – it's about having a fundamentally different way of thinking about the game. Instead of asking "How many goals did this player score?" we ask "How much value did this player create?" Instead of "How many tackles did this defender make?" we ask "How much opponent value did this defender destroy?"
        </p>

        <p>
          The mathematical frameworks we've explored – EPV, xT, OBV – aren't just academic exercises. They're practical tools that are already reshaping how the best clubs in the world find, develop, and deploy talent. When Liverpool signed Mohamed Salah from Roma for £36.9 million, traditional metrics suggested he was a good but not exceptional player. The possession value models told a different story: here was a player who consistently created value in ways that weren't being properly captured by goals and assists.
        </p>

        <p>
          The future belongs to clubs that can see value where others see noise, that can identify the players whose EPV contributions exceed their market price, and that can build teams where individual possession values combine to create something greater than the sum of their parts. In a world where everyone has access to the same traditional statistics, the competitive advantage goes to those who can read between the lines – and mathematics is showing us how.
        </p>

        <Note>
          <p>
            <strong>The Real Revolution:</strong> The mathematics of possession value doesn't just tell us who's good – it tells us <em>why</em> they're good, <em>how</em> they're good, and most importantly, whether they'll still be good in your system, with your teammates, playing your style of football. That's not just analytics; that's intelligence.
          </p>
        </Note>
      </section>
    </>
  );
};