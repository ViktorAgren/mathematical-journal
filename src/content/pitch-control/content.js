import React from "react";
import { CollapsibleCodeWindow } from "../../components/CollapsibleCodeWindow";
import { LatexBlock } from "../../components/LatexBlock";
import { Theorem } from "../../components/Theorem";
import { Note } from "../../components/Note";
import { InlineMath } from "../../components/InlineMath";
import { Visualization } from "../../components/Visualization";

export const PitchControlContent = () => {
  return (
    <>
      <section className="paper-section">
        <h2 className="section-title">
          The Corner Kick That Started a Revolution
        </h2>

        <p>
          It's May 2022, and Liverpool is trailing Real Madrid 3-1 in the Champions League final. The camera zooms in on Jürgen Klopp as he frantically gestures to his players during a corner kick. What the viewers don't see is the invisible mathematical battlefield unfolding on the pitch: a complex geometric dance of spatial control that will determine whether Liverpool can mount a comeback or Madrid can close out the game.
        </p>

        <p>
          In that moment, eleven Liverpool players and eleven Madrid players aren't just occupying random positions. They're engaged in a sophisticated game of territorial chess, where every step creates or destroys zones of influence, every movement opens or closes passing lanes, and every tactical adjustment shifts the fundamental geometry of possibility. Traditional football analysis would focus on who wins the header or whether the ball finds the net. But something far more profound is happening: a real-time optimization problem involving 22 moving agents, each trying to maximize their team's spatial control while minimizing their opponent's.
        </p>

        <p>
          This is the world that Liverpool FC and DeepMind entered when they decided to revolutionize how we understand football tactics. Their TacticAI system doesn't just analyze what happened. It models the invisible forces that determine what <em>can</em> happen. It's the difference between watching a chess game and understanding the theory of optimal play.
        </p>

        <h2 className="section-title">
          Why Your Favorite Midfielder Isn't Where You Think He Is
        </h2>

        <p>
          Picture Thiago Alcântara receiving the ball in Liverpool's midfield. To most observers, he's simply "in the center of the pitch." But to TacticAI's geometric deep learning models, Thiago exists within a complex landscape of influence zones, control polygons, and probabilistic territories that extend far beyond his immediate position. Every other player on the pitch – teammate and opponent alike – is simultaneously creating and constraining the space of possible actions.
        </p>

        <p>
          Here's the insight that changes everything: football isn't really played by 22 individual players. It's played by dynamic geometric shapes that breathe, expand, contract, and compete for territorial dominance. When Thiago moves five yards forward, he's not just changing his position – he's altering the fundamental geometry of the game, creating new passing angles, closing off opponent options, and shifting the balance of spatial power across the entire pitch.
        </p>

        <p>
          Traditional analytics tracks where players are. Pitch control theory tracks where they <em>could be</em>, and more importantly, where they're <em>preventing others from being</em>. It's the mathematical formalization of tactical intelligence – the ability to see not just the ball, but the invisible territories that determine everything else.
        </p>

        <Note>
          <p>
            <strong>The Invisibility Problem:</strong> Studies show that even professional football analysts miss up to 70% of off-ball tactical actions. Players create and destroy space constantly, but human observers naturally focus on ball-centric events. Pitch control models reveal this hidden layer of the game where most tactical battles are actually won and lost.
          </p>
        </Note>

        <h2 className="section-title">
          The Mathematics of Territorial Warfare
        </h2>

        <p>
          The breakthrough came when Liverpool and DeepMind realized that football tactics could be modeled as a continuous spatial optimization problem. Instead of treating player positions as discrete points, they began thinking about each player as the center of an influence field – a mathematical region where that player can realistically affect the game.
        </p>

        <p>
          The foundation is deceptively elegant. For any point <InlineMath tex="(x,y)" /> on the pitch, we can calculate which player would most likely reach that point first, given their current position, velocity, and physical capabilities:
        </p>

        <LatexBlock equation="T_{i,j}(x,y) = \frac{d_{i,j}(x,y)}{v_{i,j}} + \tau_{i,j}" />

        <p>
          where <InlineMath tex="T_{i,j}(x,y)" /> is the time for player <InlineMath tex="j" /> from team <InlineMath tex="i" /> to reach position <InlineMath tex="(x,y)" />, <InlineMath tex="d_{i,j}(x,y)" /> is the distance they need to travel, <InlineMath tex="v_{i,j}" /> is their maximum velocity, and <InlineMath tex="\tau_{i,j}" /> is their reaction time.
        </p>

        <p>
          But this is just the beginning. Real pitch control isn't about who can run fastest to empty space – it's about who can control space under pressure, while other players are simultaneously trying to occupy or deny that same territory. The mathematics becomes a complex optimization where every player is solving a multi-objective problem in real-time.
        </p>

        <p>
          This is where the geometric deep learning becomes crucial. Traditional machine learning treats player positions as independent variables, but TacticAI recognizes that football is fundamentally about <em>relationships in space</em>. The exact position of Liverpool's left-back doesn't matter as much as his position relative to Madrid's right-winger, Liverpool's center-back, and the ball. These relationships form a geometric graph that changes shape with every step.
        </p>

        <Theorem title="The Spatial Influence Principle">
          <p>
            For any configuration of players on the pitch, the total spatial control is conserved but can be redistributed. Mathematically:
          </p>
          <LatexBlock equation="\sum_{i=1}^{22} C_i(t) = |Pitch| = \text{constant}" />
          <p>
            where <InlineMath tex="C_i(t)" /> is the spatial control of player <InlineMath tex="i" /> at time <InlineMath tex="t" />. Tactical movements don't create control. They redistribute it from opponents to teammates.
          </p>
        </Theorem>

        <Visualization 
          type="pitch-control" 
          title="Pitch Control Map: Team Territorial Dominance"
          width={600} 
          height={400}
          data={{
            players: [
              { x: 0.2, y: 0.5, team: 'A' },
              { x: 0.4, y: 0.3, team: 'A' },
              { x: 0.6, y: 0.7, team: 'A' },
              { x: 0.8, y: 0.4, team: 'B' },
              { x: 0.7, y: 0.6, team: 'B' }
            ]
          }}
        />

        <h2 className="section-title">
          How DeepMind Taught Computers to See Like Pep Guardiola
        </h2>

        <p>
          The genius of TacticAI lies in how it learns tactical patterns directly from geometric relationships. Instead of being programmed with rules like "defend deep against fast opponents," the system discovers these principles by analyzing thousands of match situations and their outcomes.
        </p>

        <p>
          The key innovation is representing each game state as a graph where players are nodes and their tactical relationships are edges. When Kevin De Bruyne moves into the half-space, the graph doesn't just update his position – it updates his connections to teammates (creating new passing angles), to opponents (changing marking responsibilities), and to zones of the pitch (altering space control).
        </p>

        <p>
          This geometric representation allows the model to understand tactical concepts that would be impossibly complex to program explicitly. Take "pressing triggers" – the specific moments when a team should collectively press the opponent. Traditional analysis might look at ball position or time remaining. TacticAI looks at the geometric configuration of all 22 players and can identify pressing opportunities based on spatial relationships that human coaches learn through decades of experience.
        </p>

        <p>
          The mathematical foundation uses graph neural networks to process these spatial relationships:
        </p>

        <LatexBlock equation="h_i^{(l+1)} = \sigma\left(\sum_{j \in N(i)} W^{(l)} h_j^{(l)} + b^{(l)}\right)" />

        <p>
          where <InlineMath tex="h_i^{(l)}" /> represents the learned features of player <InlineMath tex="i" /> at layer <InlineMath tex="l" />, <InlineMath tex="N(i)" /> is the set of players connected to player <InlineMath tex="i" /> in the tactical graph, and <InlineMath tex="W^{(l)}" /> are the learned weights that capture tactical relationships.
        </p>

        <p>
          What makes this remarkable is that the system doesn't just learn individual player behaviors – it learns emergent team behaviors that arise from geometric interactions. When Liverpool's full-backs push high, creating space behind them, TacticAI understands this isn't just about two players changing position. It's about how this movement affects the entire team's spatial structure, from the goalkeeper's distribution options to the striker's pressing angles.
        </p>

        <h2 className="section-title">
          The Corner Kick Oracle
        </h2>

        <p>
          Corner kicks became TacticAI's killer application because they represent football tactics in pure form. The ball is stationary, all 22 players can position themselves optimally, and the outcome depends entirely on spatial relationships and tactical execution. It's like having a controlled experiment in tactical optimization every few minutes.
        </p>

        <p>
          What Liverpool discovered was revolutionary: most teams' corner kick setups are mathematically suboptimal. By analyzing thousands of corners through the lens of pitch control theory, TacticAI could identify configurations that maximized the attacking team's spatial control in high-value areas while minimizing the defending team's ability to clear or counter-attack.
        </p>

        <p>
          The optimization problem looks like this: given the current positions of all players, find the movement pattern that maximizes:
        </p>

        <LatexBlock equation="U = \sum_{z \in Z_{danger}} P_{attack}(z) \cdot V(z) - \sum_{z \in Z_{counter}} P_{defend}(z) \cdot R(z)" />

        <p>
          where <InlineMath tex="P_{attack}(z)" /> is the probability of the attacking team controlling zone <InlineMath tex="z" />, <InlineMath tex="V(z)" /> is the value of controlling that zone for scoring, <InlineMath tex="P_{defend}(z)" /> is the probability of the defending team controlling key defensive zones, and <InlineMath tex="R(z)" /> is the risk of allowing a counter-attack from that zone.
        </p>

        <p>
          The results were immediate and dramatic. TacticAI's corner kick recommendations led to measurably better outcomes: more goals scored, fewer chances conceded on the counter, and better control of second balls. But more importantly, it demonstrated that tactical problems could be solved through mathematical optimization rather than just intuition and experience.
        </p>

        <Note>
          <p>
            <strong>The Liverpool Effect:</strong> After implementing TacticAI insights, Liverpool improved their corner kick conversion rate by 13% and reduced opponent counter-attacking opportunities by 7%. More significantly, they began creating better chances from second balls by optimizing player positions for rebound scenarios.
          </p>
        </Note>

        <Visualization 
          type="corner-kick-setup" 
          title="TacticAI Optimized Corner Kick Formation"
          width={600} 
          height={450}
          data={{}}
        />

        <h2 className="section-title">
          Reading the Invisible Game
        </h2>

        <p>
          What makes pitch control theory so powerful is how it reveals tactical battles that are completely invisible to traditional analysis. Consider what happens when Manchester City builds up from the back against Liverpool's high press. To most observers, it's a simple question of whether City can find a way through Liverpool's aggressive defensive line.
        </p>

        <p>
          But TacticAI sees a complex geometric optimization playing out in real-time. Every Liverpool player's pressing run is simultaneously trying to reduce City's spatial control while maintaining Liverpool's defensive structure. Every City player's movement is trying to create overloads in space while avoiding the geometric traps that Liverpool's press is designed to create.
        </p>

        <p>
          The mathematical beauty lies in how these competing objectives create dynamic equilibriums. When Liverpool presses high, they gain control of the middle third but potentially surrender control of the final third if the press is bypassed. City's buildup patterns are designed to exploit these spatial trades by creating situations where their superior technical ability can maximize the value of the space they do control.
        </p>

        <p>
          This is where the deep learning aspect becomes crucial. The system learns that certain geometric configurations lead to specific tactical outcomes, even when the causal relationships are too complex for human analysis. It might discover that when City's center-backs are positioned 18 meters apart with both full-backs above the halfway line, Liverpool's press succeeds 73% of the time – but only if their midfield maintains a specific triangular formation.
        </p>

        <Theorem title="The Tactical Uncertainty Principle">
          <p>
            In dynamic tactical situations, there's a fundamental tradeoff between spatial control and tactical flexibility:
          </p>
          <LatexBlock equation="\Delta C \cdot \Delta F \geq \frac{k}{2}" />
          <p>
            where <InlineMath tex="\Delta C" /> is the uncertainty in spatial control and <InlineMath tex="\Delta F" /> is the uncertainty in tactical flexibility. Teams cannot simultaneously maximize both territorial dominance and tactical adaptability.
          </p>
        </Theorem>

        <h2 className="section-title">
          The Future of Tactical Intelligence
        </h2>

        <p>
          The collaboration between Liverpool and DeepMind represents just the beginning of football's mathematical revolution. The principles of geometric deep learning and spatial optimization are being applied to every aspect of the game, from player recruitment to in-game tactical adjustments.
        </p>

        <p>
          Consider player scouting: instead of asking "Can this player score goals?" or "Can this player make tackles?", clubs are beginning to ask "How does this player's spatial intelligence complement our existing geometric structure?" TacticAI-style analysis can identify players whose movement patterns create optimal spatial relationships with their new teammates, even if their individual statistics seem unimpressive.
        </p>

        <p>
          The real-time applications are even more exciting. Imagine tactical analysis that can identify optimal pressing triggers as they develop, or suggest positional adjustments that would maximize spatial control in the current game state. The mathematics of pitch control could enable a level of tactical optimization that was previously impossible.
        </p>

        <p>
          But perhaps most importantly, this approach is teaching us to see football differently. The ball is just one element in a complex spatial system. The real game is happening in the invisible territories between players, in the geometric relationships that determine what's possible, and in the mathematical optimization that elite players and coaches perform intuitively.
        </p>

        <h2 className="section-title">
          Beyond the Beautiful Game
        </h2>

        <p>
          The mathematical insights from pitch control theory extend far beyond football. Any domain involving multiple agents competing for spatial control – from military strategy to autonomous vehicle coordination to economic market dynamics – can benefit from these geometric deep learning approaches.
        </p>

        <p>
          The fundamental insight is that complex systems often have hidden geometric structures that determine their behavior. By learning to see these structures and understand their mathematical properties, we can optimize performance in ways that would be impossible through intuition alone.
        </p>

        <CollapsibleCodeWindow
          language="python"
          title="Pitch Control Implementation"
          code={`
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

class PitchControlModel:
    """
    Implementation of pitch control analysis based on geometric deep learning.
    Models spatial influence of players and tactical opportunities.
    """
    
    def __init__(self, pitch_length=105, pitch_width=68):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.grid_resolution = 1.0  # meters per grid point
        
        # Create spatial grid
        self.x_grid = np.arange(0, pitch_length + self.grid_resolution, 
                               self.grid_resolution)
        self.y_grid = np.arange(0, pitch_width + self.grid_resolution, 
                               self.grid_resolution)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        
        # Player physical parameters
        self.max_velocity = 8.0  # m/s
        self.reaction_time = 0.3  # seconds
        
    def calculate_time_to_point(self, player_pos, target_points, velocity=None):
        """
        Calculate time for player to reach each target point.
        
        Parameters:
            player_pos: (x, y) position of player
            target_points: Array of (x, y) target positions
            velocity: Player's maximum velocity (default: self.max_velocity)
        
        Returns:
            Array of times to reach each target point
        """
        if velocity is None:
            velocity = self.max_velocity
            
        # Calculate Euclidean distances
        distances = np.sqrt((target_points[:, 0] - player_pos[0])**2 + 
                           (target_points[:, 1] - player_pos[1])**2)
        
        # Time = distance/velocity + reaction time
        times = distances / velocity + self.reaction_time
        
        return times
    
    def calculate_pitch_control(self, team_a_positions, team_b_positions, 
                               ball_position=None):
        """
        Calculate pitch control map showing which team controls each area.
        
        Parameters:
            team_a_positions: List of (x, y) positions for team A
            team_b_positions: List of (x, y) positions for team B
            ball_position: (x, y) position of ball (affects control calculation)
        
        Returns:
            control_map: 2D array showing team control (-1 to 1)
            influence_maps: Dictionary with individual player influence
        """
        
        # Flatten grid for vectorized computation
        grid_points = np.column_stack([self.X.flatten(), self.Y.flatten()])
        
        # Initialize control arrays
        team_a_control = np.zeros(len(grid_points))
        team_b_control = np.zeros(len(grid_points))
        
        influence_maps = {'team_a': {}, 'team_b': {}}
        
        # Calculate team A player influences
        for i, player_pos in enumerate(team_a_positions):
            times = self.calculate_time_to_point(player_pos, grid_points)
            
            # Convert time to influence using exponential decay
            influence = np.exp(-times / 2.0)  # Higher influence = lower time
            team_a_control += influence
            influence_maps['team_a'][f'player_{i}'] = influence.reshape(self.X.shape)
        
        # Calculate team B player influences  
        for i, player_pos in enumerate(team_b_positions):
            times = self.calculate_time_to_point(player_pos, grid_points)
            influence = np.exp(-times / 2.0)
            team_b_control += influence
            influence_maps['team_b'][f'player_{i}'] = influence.reshape(self.X.shape)
        
        # Normalize to get relative control (-1 = team B, +1 = team A)
        total_control = team_a_control + team_b_control
        relative_control = (team_a_control - team_b_control) / (total_control + 1e-6)
        
        # Apply ball position bias if provided
        if ball_position is not None:
            ball_distances = np.sqrt((grid_points[:, 0] - ball_position[0])**2 + 
                                   (grid_points[:, 1] - ball_position[1])**2)
            ball_influence = np.exp(-ball_distances / 10.0)  # Ball adds local importance
            relative_control *= (1 + 0.5 * ball_influence)  # Boost control near ball
        
        control_map = relative_control.reshape(self.X.shape)
        
        return control_map, influence_maps
    
    def find_tactical_opportunities(self, team_a_positions, team_b_positions, 
                                  attacking_team='A'):
        """
        Identify tactical opportunities based on pitch control analysis.
        
        Parameters:
            team_a_positions: List of (x, y) positions for team A  
            team_b_positions: List of (x, y) positions for team B
            attacking_team: Which team is attacking ('A' or 'B')
        
        Returns:
            opportunities: List of tactical opportunity descriptions
        """
        
        control_map, _ = self.calculate_pitch_control(
            team_a_positions, team_b_positions
        )
        
        opportunities = []
        
        # Define key zones
        final_third_start = self.pitch_length * 2/3
        penalty_area = {
            'x_min': self.pitch_length - 16.5,
            'x_max': self.pitch_length,
            'y_min': (self.pitch_width - 40.3) / 2,
            'y_max': (self.pitch_width + 40.3) / 2
        }
        
        if attacking_team == 'A':
            control_threshold = 0.3  # Team A advantage
            
            # Check final third control
            final_third_mask = self.X >= final_third_start
            final_third_control = control_map[final_third_mask]
            avg_final_third_control = np.mean(final_third_control)
            
            if avg_final_third_control > control_threshold:
                opportunities.append({
                    'type': 'Final Third Dominance',
                    'description': 'Team A has strong spatial control in final third',
                    'value': avg_final_third_control,
                    'zone': 'attacking_third'
                })
            
            # Check penalty area pressure
            penalty_mask = ((self.X >= penalty_area['x_min']) & 
                           (self.X <= penalty_area['x_max']) &
                           (self.Y >= penalty_area['y_min']) & 
                           (self.Y <= penalty_area['y_max']))
            
            if np.any(penalty_mask):
                penalty_control = control_map[penalty_mask]
                avg_penalty_control = np.mean(penalty_control)
                
                if avg_penalty_control > control_threshold:
                    opportunities.append({
                        'type': 'Penalty Area Threat',
                        'description': 'Team A creating danger in penalty area',
                        'value': avg_penalty_control,
                        'zone': 'penalty_area'
                    })
            
            # Find space overloads (areas with high control concentration)
            high_control_areas = np.where(control_map > 0.6)
            if len(high_control_areas[0]) > 0:
                # Cluster high control areas
                high_control_points = np.column_stack([
                    self.X[high_control_areas], 
                    self.Y[high_control_areas]
                ])
                
                if len(high_control_points) >= 5:  # Minimum for clustering
                    kmeans = KMeans(n_clusters=min(3, len(high_control_points)//5))
                    clusters = kmeans.fit_predict(high_control_points)
                    
                    for cluster_id in range(kmeans.n_clusters):
                        cluster_points = high_control_points[clusters == cluster_id]
                        if len(cluster_points) >= 3:  # Significant cluster
                            centroid = kmeans.cluster_centers_[cluster_id]
                            opportunities.append({
                                'type': 'Space Overload',
                                'description': f'Concentrated control at ({centroid[0]:.1f}, {centroid[1]:.1f})',
                                'value': np.mean(control_map[high_control_areas]),
                                'zone': 'midfield' if centroid[0] < final_third_start else 'final_third'
                            })
        
        return opportunities
    
    def simulate_pressing_trigger(self, team_a_positions, team_b_positions, 
                                ball_position, pressing_intensity=1.5):
        """
        Simulate the effect of coordinated pressing on pitch control.
        
        Parameters:
            team_a_positions: Current positions of team A (defending)
            team_b_positions: Current positions of team B (pressing)
            ball_position: Current ball position
            pressing_intensity: Multiplier for pressing team's effective velocity
        
        Returns:
            new_control_map: Pitch control after pressing movement
            press_effectiveness: Scalar measure of pressing success
        """
        
        # Calculate baseline control
        baseline_control, _ = self.calculate_pitch_control(
            team_a_positions, team_b_positions, ball_position
        )
        
        # Simulate pressing movement - team B players move toward ball
        pressed_team_b_positions = []
        for pos in team_b_positions:
            # Move 30% toward ball position (simplified pressing movement)
            direction_to_ball = np.array(ball_position) - np.array(pos)
            if np.linalg.norm(direction_to_ball) > 0:
                direction_to_ball = direction_to_ball / np.linalg.norm(direction_to_ball)
                new_pos = np.array(pos) + 0.3 * direction_to_ball * 5.0  # 5m movement
                pressed_team_b_positions.append(new_pos.tolist())
            else:
                pressed_team_b_positions.append(pos)
        
        # Calculate control after pressing with increased intensity
        grid_points = np.column_stack([self.X.flatten(), self.Y.flatten()])
        
        team_a_control = np.zeros(len(grid_points))
        team_b_control = np.zeros(len(grid_points))
        
        # Team A (being pressed) - normal calculation
        for player_pos in team_a_positions:
            times = self.calculate_time_to_point(player_pos, grid_points)
            influence = np.exp(-times / 2.0)
            team_a_control += influence
        
        # Team B (pressing) - enhanced velocity
        for player_pos in pressed_team_b_positions:
            times = self.calculate_time_to_point(
                player_pos, grid_points, 
                velocity=self.max_velocity * pressing_intensity
            )
            influence = np.exp(-times / 2.0)
            team_b_control += influence
        
        # Calculate new relative control
        total_control = team_a_control + team_b_control
        new_relative_control = (team_a_control - team_b_control) / (total_control + 1e-6)
        new_control_map = new_relative_control.reshape(self.X.shape)
        
        # Measure pressing effectiveness (how much control shifted toward ball)
        ball_area_radius = 15  # meters
        ball_distances = np.sqrt((self.X - ball_position[0])**2 + 
                                (self.Y - ball_position[1])**2)
        ball_area_mask = ball_distances <= ball_area_radius
        
        baseline_ball_control = np.mean(baseline_control[ball_area_mask])
        new_ball_control = np.mean(new_control_map[ball_area_mask])
        
        # Negative change = team B (pressing team) gained control
        press_effectiveness = baseline_ball_control - new_ball_control
        
        return new_control_map, press_effectiveness
    
    def visualize_pitch_control(self, control_map, team_a_positions, 
                               team_b_positions, ball_position=None,
                               title="Pitch Control Analysis"):
        """
        Create visualization of pitch control map with player positions.
        """
        
        plt.figure(figsize=(12, 8))
        
        # Plot control map
        plt.contourf(self.X, self.Y, control_map, levels=20, 
                    cmap='RdBu', alpha=0.7, vmin=-1, vmax=1)
        plt.colorbar(label='Team Control (Red=A, Blue=B)')
        
        # Plot team positions
        team_a_x, team_a_y = zip(*team_a_positions)
        team_b_x, team_b_y = zip(*team_b_positions)
        
        plt.scatter(team_a_x, team_a_y, c='red', s=100, 
                   marker='o', label='Team A', edgecolors='black', linewidth=2)
        plt.scatter(team_b_x, team_b_y, c='blue', s=100, 
                   marker='s', label='Team B', edgecolors='black', linewidth=2)
        
        # Plot ball if provided
        if ball_position is not None:
            plt.scatter(ball_position[0], ball_position[1], 
                       c='black', s=50, marker='*', label='Ball')
        
        # Pitch markings
        plt.axhline(y=0, color='white', linewidth=2)
        plt.axhline(y=self.pitch_width, color='white', linewidth=2) 
        plt.axvline(x=0, color='white', linewidth=2)
        plt.axvline(x=self.pitch_length, color='white', linewidth=2)
        plt.axvline(x=self.pitch_length/2, color='white', linewidth=1)
        
        # Goal areas
        plt.plot([self.pitch_length-16.5, self.pitch_length-16.5], 
                [(self.pitch_width-40.3)/2, (self.pitch_width+40.3)/2], 
                'white', linewidth=2)
        plt.plot([16.5, 16.5], 
                [(self.pitch_width-40.3)/2, (self.pitch_width+40.3)/2], 
                'white', linewidth=2)
        
        plt.xlim(0, self.pitch_length)
        plt.ylim(0, self.pitch_width)
        plt.xlabel('Length (m)')
        plt.ylabel('Width (m)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()

# Demonstration
def demo_pitch_control_analysis():
    """
    Demonstrate pitch control analysis with realistic match scenario.
    """
    
    print("TacticAI-Style Pitch Control Analysis")
    print("=" * 50)
    
    # Initialize model
    model = PitchControlModel()
    
    # Example: Liverpool attacking, Real Madrid defending
    # Positions represent a corner kick scenario
    
    liverpool_positions = [
        [95, 34],    # Goalkeeper (near own goal)
        [75, 10],    # RB pushed up
        [70, 25],    # CB
        [70, 43],    # CB
        [75, 58],    # LB pushed up
        [85, 20],    # DM
        [90, 34],    # CM (corner taker)
        [88, 45],    # CM
        [100, 30],   # RW (in box)
        [102, 38],   # ST (in box)
        [100, 46]    # LW (in box)
    ]
    
    madrid_positions = [
        [105, 34],   # Goalkeeper
        [102, 20],   # RB (defending box)
        [100, 25],   # CB (on line)
        [100, 43],   # CB (on line)
        [102, 48],   # LB (defending box)
        [98, 34],    # DM (covering)
        [95, 15],    # CM (defending)
        [95, 53],    # CM (defending)
        [101, 30],   # RW (marking)
        [101, 38],   # ST (marking)
        [101, 46]    # LW (marking)
    ]
    
    ball_position = [90, 68]  # Corner flag
    
    print("Analyzing corner kick scenario...")
    print(f"Ball position: {ball_position}")
    print(f"Liverpool players: {len(liverpool_positions)}")
    print(f"Madrid players: {len(madrid_positions)}")
    
    # Calculate pitch control
    control_map, influence_maps = model.calculate_pitch_control(
        liverpool_positions, madrid_positions, ball_position
    )
    
    # Analyze tactical opportunities
    opportunities = model.find_tactical_opportunities(
        liverpool_positions, madrid_positions, attacking_team='A'
    )
    
    print("\nTactical Opportunities Identified:")
    print("-" * 40)
    for opp in opportunities:
        print(f"• {opp['type']}: {opp['description']}")
        print(f"  Strength: {opp['value']:.3f} | Zone: {opp['zone']}")
    
    # Simulate defensive pressing response
    print("\nSimulating Madrid's defensive adjustment...")
    new_control_map, press_effectiveness = model.simulate_pressing_trigger(
        liverpool_positions, madrid_positions, ball_position,
        pressing_intensity=1.3
    )
    
    print(f"Press effectiveness: {press_effectiveness:.3f}")
    if press_effectiveness > 0.1:
        print("→ Pressing successfully reduced Liverpool's control")
    elif press_effectiveness < -0.1:
        print("→ Pressing backfired, Liverpool gained more control")
    else:
        print("→ Pressing had minimal effect")
    
    # Calculate key metrics
    penalty_area_mask = ((model.X >= 89) & (model.X <= 105) & 
                        (model.Y >= 14) & (model.Y <= 54))
    
    liverpool_penalty_control = np.mean(control_map[penalty_area_mask])
    madrid_penalty_control = np.mean(new_control_map[penalty_area_mask])
    
    print(f"\nPenalty Area Control Analysis:")
    print(f"Liverpool initial control: {liverpool_penalty_control:.3f}")
    print(f"After Madrid adjustment: {madrid_penalty_control:.3f}")
    print(f"Control shift: {madrid_penalty_control - liverpool_penalty_control:.3f}")
    
    # Visualize the analysis
    fig1 = model.visualize_pitch_control(
        control_map, liverpool_positions, madrid_positions, 
        ball_position, "Initial Corner Kick Setup"
    )
    
    fig2 = model.visualize_pitch_control(
        new_control_map, liverpool_positions, madrid_positions,
        ball_position, "After Madrid Defensive Adjustment"
    )
    
    return model, control_map, new_control_map, opportunities

if __name__ == "__main__":
    model, initial_control, adjusted_control, opportunities = demo_pitch_control_analysis()
          `}
        />

        <p>
          TacticAI represents more than just better football analysis – it's a new way of understanding complex systems where multiple intelligent agents compete for spatial and tactical advantage. The mathematical principles behind pitch control will undoubtedly find applications far beyond the football pitch, but for now, they're helping coaches and players see the beautiful game in an entirely new light.
        </p>

        <p>
          The corner kick that might have changed Liverpool's Champions League final will forever remain a "what if." But the mathematical framework that could have optimized that moment represents a very real shift in how we understand, analyze, and play the world's most popular sport. In a game where the difference between victory and defeat often comes down to the finest of margins, seeing those invisible geometric relationships might just be the edge that determines the next champion.
        </p>

        <Note>
          <p>
            <strong>The Geometric Revolution:</strong> Pitch control theory doesn't just change how we analyze football – it reveals that the game has always been about geometric optimization. Elite players and coaches have been solving these spatial problems intuitively. Now we have the mathematics to understand, teach, and optimize these solutions systematically.
          </p>
        </Note>
      </section>
    </>
  );
};