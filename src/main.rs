use ac_library::dsu;
use proconio::input;
use rand::seq::SliceRandom;

const TIME_LIMIT: f64 = 1.9;
const MAX_LINE: usize = 400000;
const MAX_VERTICES: usize = 1000;
const NUM_GRID: usize = 50;
const HEIGHT: usize = 100000;
const WIDTH: usize = 100000;

// x, y
const DIR: [(i32, i32); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

/**
 * 50x50のグリッドで考える
 * 1. 一番鯖が多いマスを探す
 * 2. 他のグリッドまでの経路で鯖が最も多い経路を探す（イワシをなるべく取らないような経路）。
 * 3. エリアの任意の点から別のエリアの任意の点までの最適な経路を調べる。
 * 4. 経路を増やせなくなったらエリアを拡大・縮小して最適化する。
 */
fn main() {
    get_time();
    let input = read_input();
    let output = solve(input);
    print_output(output);
}

struct Point {
    x: i32,
    y: i32,
}

struct Input {
    n: usize,
    saba: Vec<Point>,
    iwashi: Vec<Point>,
}

fn read_input() -> Input {
    input! {
        n: usize,
        points: [(i32, i32); 2 * n],
    }

    Input {
        n,
        saba: points[0..n].iter().map(|&(x, y)| Point { x, y }).collect(),
        iwashi: points[n..].iter().map(|&(x, y)| Point { x, y }).collect(),
    }
}

struct Output {
    m: usize,
    points: Vec<Point>,
}

fn print_output(output: Output) {
    println!("{}", output.m);
    for point in output.points {
        println!("{} {}", point.x, point.y);
    }
}

/**
 * 使用したグリッドから頂点と辺数を計算する
 * @return (頂点, 辺数)
 */
fn calc_points_from_used_grid(grid: &Vec<Vec<bool>>) -> (Vec<Point>, usize) {
    // (1,1) -> (2000, 2000)とする。
    // (1, 2) -> (2000, 4000)とする。
    // (51, 51) -> (100000, 100000)とする。
    // 0 -> 右から
    // 1 -> 下から
    // 2 -> 左から
    // 3 -> 上から
    // 反時計回りを考える

    let mut points_in = vec![vec![vec![0; 4]; NUM_GRID + 1]; NUM_GRID + 1];
    for x in 0..NUM_GRID {
        for y in 0..NUM_GRID {
            if grid[x][y] {
                points_in[x][y][0] += 1;
                points_in[x][y][1] -= 1;
                points_in[x + 1][y][1] += 1;
                points_in[x + 1][y][2] -= 1;
                points_in[x + 1][y + 1][2] += 1;
                points_in[x + 1][y + 1][3] -= 1;
                points_in[x][y + 1][3] += 1;
                points_in[x][y + 1][0] -= 1;
            }
        }
    }

    let mut points = vec![];
    let mut points_set = std::collections::HashSet::new();
    let mut edges = vec![vec![]; (NUM_GRID + 1) * (NUM_GRID + 1)];
    let mut num_edges = 0;
    for x in 0..NUM_GRID + 1 {
        for y in 0..NUM_GRID + 1 {
            if points_in[x][y][0] != 0 {
                edges[x * (NUM_GRID + 1) + y].push((x + 1) * (NUM_GRID + 1) + y);
                num_edges += 1;
            }
            if points_in[x][y][1] != 0 {
                edges[x * (NUM_GRID + 1) + y].push(x * (NUM_GRID + 1) + y + 1);
                num_edges += 1;
            }
            if points_in[x][y][2] != 0 {
                edges[x * (NUM_GRID + 1) + y].push((x - 1) * (NUM_GRID + 1) + y);
                num_edges += 1;
            }
            if points_in[x][y][3] != 0 {
                edges[x * (NUM_GRID + 1) + y].push(x * (NUM_GRID + 1) + y - 1);
                num_edges += 1;
            }

            if points_in[x][y][0] != 0
                || points_in[x][y][1] != 0
                || points_in[x][y][2] != 0
                || points_in[x][y][3] != 0
            {
                points.push((x, y));
                points_set.insert((x, y));
            }
        }
    }

    // 頂点を並び替える
    let mut start = points[0];
    let mut points_ret = vec![start];
    let mut visited = std::collections::HashSet::new();

    loop {
        visited.insert(start);
        let index = start.0 * (NUM_GRID + 1) + start.1;
        let mut next = (1000, 1000);
        for nx in &edges[index] {
            let x = nx / (NUM_GRID + 1);
            let y = nx % (NUM_GRID + 1);
            if !visited.contains(&(x, y)) {
                next = (x, y);
                points_ret.push(next);
                break;
            }
        }

        if next == (1000, 1000) {
            break;
        }
        start = next;
    }

    (
        points_ret
            .into_iter()
            .map(|(x, y)| Point {
                x: 2000 * x as i32,
                y: 2000 * y as i32,
            })
            .collect(),
        num_edges / 2,
    )
}

fn solve(input: Input) -> Output {
    let mut num_diff_grid: Vec<Vec<i32>> = vec![vec![0; NUM_GRID]; NUM_GRID];
    let area_width = WIDTH / NUM_GRID;

    for i in 0..input.n {
        let x = input.saba[i].x / area_width as i32;
        let y = input.saba[i].y / area_width as i32;
        num_diff_grid[x as usize][y as usize] += 1;
    }

    for i in 0..input.n {
        let x = input.iwashi[i].x / area_width as i32;
        let y = input.iwashi[i].y / area_width as i32;
        num_diff_grid[x as usize][y as usize] -= 1;
    }

    let mut max_grid_index = (0, 0);
    let mut max_num = 0;
    for i in 0..NUM_GRID {
        for j in 0..NUM_GRID {
            if num_diff_grid[i][j] > max_num {
                max_num = num_diff_grid[i][j];
                max_grid_index = (i, j);
            }
        }
    }

    // エリアを伸ばす経路を探す
    let mut prev = vec![vec![(-1, -1); NUM_GRID]; NUM_GRID];
    let mut amount = vec![vec![-1000000; NUM_GRID]; NUM_GRID];
    prev[max_grid_index.0][max_grid_index.1] = (max_grid_index.0 as i32, max_grid_index.1 as i32);
    amount[max_grid_index.0][max_grid_index.1] = max_num;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(max_grid_index);
    while let Some((x, y)) = queue.pop_front() {
        for i in 0..4 {
            let nx = x as i32 + DIR[i].0;
            let ny = y as i32 + DIR[i].1;
            if nx < 0 || ny < 0 || nx >= NUM_GRID as i32 || ny >= NUM_GRID as i32 {
                continue;
            }
            if prev[nx as usize][ny as usize] != (-1, -1) {
                continue;
            }
            if amount[x][y] + num_diff_grid[nx as usize][ny as usize]
                > amount[nx as usize][ny as usize]
            {
                amount[nx as usize][ny as usize] =
                    amount[x][y] + num_diff_grid[nx as usize][ny as usize];
                prev[nx as usize][ny as usize] = (x as i32, y as i32);
                queue.push_back((nx as usize, ny as usize));
            }
        }
    }

    eprintln!("spanning tree done");

    let mut max_amount = 0;
    let mut max_amount_index = (0, 0);
    for i in 0..NUM_GRID {
        for j in 0..NUM_GRID {
            if amount[i][j] > max_amount {
                max_amount = amount[i][j];
                max_amount_index = (i, j);
            }
        }
    }

    let mut used_grid = vec![vec![false; NUM_GRID]; NUM_GRID];
    let mut x = max_amount_index.0;
    let mut y = max_amount_index.1;
    while prev[x][y] != (x as i32, y as i32) {
        used_grid[x][y] = true;
        let (nx, ny) = prev[x][y];
        x = nx as usize;
        y = ny as usize;
    }

    let (point_tmp, edge_tmp) = calc_points_from_used_grid(&used_grid);
    let mut cur_length = edge_tmp * 2000;
    let mut cur_vertices = point_tmp.len();

    let mut output = Output {
        m: cur_vertices,
        points: point_tmp,
    };
    let mut max_not_over = 5;
    let mut cur_score = 0;
    let mut best_score = cur_score;
    for i in 0..NUM_GRID {
        for j in 0..NUM_GRID {
            if used_grid[i][j] {
                cur_score += num_diff_grid[i][j];
            }
        }
    }

    let mut best_grid = used_grid.clone();
    let mut initial_used_grid = used_grid.clone();

    let mut num_not_over = 0;
    let mut num_iter = 0;
    while get_time() < 1.9 {
        num_iter += 1;
        let mut adj: Vec<(usize, usize)> = Vec::new();
        for x in 0..NUM_GRID {
            for y in 0..NUM_GRID {
                if used_grid[x][y] {
                    for i in 0..4 {
                        let nx = x as i32 + DIR[i].0;
                        let ny = y as i32 + DIR[i].1;
                        if nx < 0 || ny < 0 || nx >= NUM_GRID as i32 || ny >= NUM_GRID as i32 {
                            continue;
                        }
                        if used_grid[nx as usize][ny as usize] {
                            continue;
                        }
                        adj.push((nx as usize, ny as usize));
                    }
                }
            }
        }
        adj.shuffle(&mut rand::thread_rng());

        for a in adj {
            let mut dsu_before = dsu::Dsu::new(NUM_GRID * NUM_GRID);
            for x in 0..NUM_GRID {
                for y in 0..NUM_GRID {
                    dsu_before.merge(x * NUM_GRID + y, x * NUM_GRID + y);

                    let nx1 = x as i32 + 1;
                    let ny1 = y as i32;

                    if nx1 < NUM_GRID as i32
                        && used_grid[nx1 as usize][ny1 as usize] == used_grid[x][y]
                    {
                        dsu_before.merge(x * NUM_GRID + y, nx1 as usize * NUM_GRID + ny1 as usize);
                    }

                    let nx2 = x as i32;
                    let ny2 = y as i32 + 1;

                    if ny2 < NUM_GRID as i32
                        && used_grid[nx2 as usize][ny2 as usize] == used_grid[x][y]
                    {
                        dsu_before.merge(x * NUM_GRID + y, nx2 as usize * NUM_GRID + ny2 as usize);
                    }
                }
            }

            if num_diff_grid[a.0][a.1] <= 0 {
                continue;
            }
            used_grid[a.0][a.1] = true;

            let mut dsu_after = dsu::Dsu::new(NUM_GRID * NUM_GRID);

            for x in 0..NUM_GRID {
                for y in 0..NUM_GRID {
                    dsu_after.merge(x * NUM_GRID + y, x * NUM_GRID + y);

                    let nx1 = x as i32 + 1;
                    let ny1 = y as i32;

                    if nx1 < NUM_GRID as i32
                        && used_grid[nx1 as usize][ny1 as usize] == used_grid[x][y]
                    {
                        dsu_after.merge(x * NUM_GRID + y, nx1 as usize * NUM_GRID + ny1 as usize);
                    }

                    let nx2 = x as i32;
                    let ny2 = y as i32 + 1;

                    if ny2 < NUM_GRID as i32
                        && used_grid[nx2 as usize][ny2 as usize] == used_grid[x][y]
                    {
                        dsu_after.merge(x * NUM_GRID + y, nx2 as usize * NUM_GRID + ny2 as usize);
                    }
                }
            }

            if dsu_after.groups().len() > dsu_before.groups().len() {
                used_grid[a.0][a.1] = false;
                continue;
            }

            let (points, num_edge) = calc_points_from_used_grid(&used_grid);

            if points.len() > MAX_VERTICES || num_edge * 2000 > MAX_LINE {
                used_grid[a.0][a.1] = false;
            }
        }

    }

    for y in 0..NUM_GRID {
        for x in 0..NUM_GRID {
            eprint!("{}", if used_grid[x][y] { "#" } else { "." });
        }
        eprintln!();
    }

    // 連結なグリッドから使用している頂点を列挙していく。
    let (points, num_edges) = calc_points_from_used_grid(&used_grid);

    eprintln!("calc points done");
    eprintln!("num_iter: {}", num_iter);
    Output {
        m: points.len(),
        points,
    }
}

pub fn get_time() -> f64 {
    static mut STIME: f64 = -1.0;
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    let ms = t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9;
    unsafe {
        if STIME < 0.0 {
            STIME = ms;
        }
        // ローカル環境とジャッジ環境の実行速度差はget_timeで吸収しておくと便利
        #[cfg(feature = "local")]
        {
            (ms - STIME) * 1.0
        }
        #[cfg(not(feature = "local"))]
        {
            ms - STIME
        }
    }
}
