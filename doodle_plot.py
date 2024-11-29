import matplotlib.pyplot as plt
import numpy as np

def softmax(values):
    exp_values = np.exp(values - np.max(values))  # Subtract max for numerical stability
    return exp_values / np.sum(exp_values)

def plot_drawing(data, output_file="drawing_plot.png"):

    x_prev, y_prev = None, None  
    plt.figure(figsize=(6, 6))
    plt.axis([0, 255, 0, 255])
    plt.gca().invert_yaxis() 

    for point in data:
        x, y, on_paper, termination = point

        softmax_values = softmax([on_paper, termination])
        print(on_paper)
        on_paper_prob = softmax_values[0]
        termination_prob = softmax_values[1]

        if on_paper_prob > 0.0: 
            if x_prev is not None and y_prev is not None:
                plt.plot([x_prev, x], [y_prev, y], color="black")  # Draw a line

        if termination_prob < 0.0:
            break

        x_prev, y_prev = x, y


    plt.title("Generated Doodle")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    # Save the figure to a file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    plt.close() 


# data = [[33.19672107696533, 218.24176043272018, -0.8204028010368347, -0.7529687285423279], [3.5357558727264404, 183.84706623852253, 0.7373279333114624, -0.7748672366142273], [124.7210746910423, 48.108716905117035, -0.8927876353263855, -0.8651196360588074], [110.46665590256453, 98.24147310107946, 0.22448015213012695, -0.766822338104248], [101.40329673886299, 120.44019685126841, 0.8515060544013977, -0.8299556374549866], [155.27717638760805, 69.44883920252323, 0.2679109573364258, -0.7845658659934998], [136.1643416248262, 67.48266890645027, 0.8566989898681641, -0.7742918133735657], [115.07193088531494, 88.68688009679317, -1.0, -0.8730380535125732], [120.4693161137402, 127.99189141311217, 0.8207942843437195, -0.8262119889259338], [101.14646852016449, 136.12466510385275, 0.8753582835197449, -0.8586341142654419], [158.26416950672865, 114.22434646636248, 0.8229447603225708, -0.8723497986793518], [147.36573681235313, 84.69174787402153, -0.16762052476406097, -0.8941418528556824], [132.65065639279783, 79.82902340590954, -0.21839262545108795, -0.8741228580474854], [153.2132601365447, 96.25944815576077, 0.9088683128356934, -0.8920520544052124], [166.26441411674023, 100.44318705797195, 0.8442657589912415, -0.8734880685806274], [200.10270416736603, 93.29102843999863, 0.8500511050224304, -0.9013375639915466], [206.98560923337936, 103.32326341420412, 0.8573815226554871, -0.8928782343864441], [162.95718558132648, 159.53246735036373, 0.8032117486000061, -0.8395030498504639], [156.09300021082163, 204.5262672007084, 0.8465934991836548, -0.1583372801542282], [30.71044608950615, 99.55941718071699, 0.3395181894302368, -0.4986545741558075], [16.197641640901566, 37.05631837248802, 0.2329060286283493, -0.5966725945472717], [14.073464423418045, 26.556303799152374, 0.5115069150924683, -0.5772231817245483], [24.168131947517395, 15.443230122327805, 0.4678264856338501, -0.591749906539917], [5.891295075416565, 28.113711029291153, 0.5075439810752869, -0.6364170908927917], [28.347079306840897, 28.084969371557236, 0.5996002554893494, -0.6989631056785583], [6.935752630233765, 28.199282437562943, 0.6387770175933838, -0.6241894364356995], [34.62487444281578, 8.412976562976837, 0.21280483901500702, -0.7246147394180298], [44.30077403783798, 42.32896864414215, 0.2604268491268158, -0.7893773913383484], [76.03999212384224, 42.33358159661293, 0.556057870388031, -0.7617852091789246], [73.02540868520737, 47.26260110735893, 0.6886929273605347, -0.7649556398391724], [97.35051970928907, 53.65571245551109, 0.644497275352478, -0.7870078086853027], [100.16906406730413, 48.55823278427124, 0.42524370551109314, -0.7805288434028625], [124.11993317306042, 96.08021177351475, 0.7120077610015869, -0.6430525779724121], [142.77509769424796, 42.67026633024216, 0.14760345220565796, -0.625199556350708], [167.38732986152172, 53.52506026625633, 0.7294389605522156, -0.7075307965278625], [126.34765625931323, 92.73853428661823, 0.6561369895935059, -0.7605843544006348], [64.54335182905197, 129.40303798997775, 0.48976951837539673, -0.30277520418167114], [24.37269017100334, 49.116506427526474, 0.46082496643066406, 0.00823196955025196], [8.52063238620758, 55.12320891022682, 0.26962336897850037, -0.06825186312198639], [9.671757817268372, 18.92030194401741, 0.04279081150889397, -0.09017565101385117], [8.420325368642807, 15.212453305721283, 0.08244600147008896, -0.055196117609739304], [10.764259994029999, 17.671232968568802, 0.02543693780899048, 0.018362190574407578], [9.373876601457596, 6.771472245454788, 0.026745818555355072, -0.047949496656656265], [5.806483626365662, 8.19546863436699, 0.010230155661702156, -0.06165149062871933], [7.657691091299057, 4.398431181907654, -0.12233374267816544, -0.08228809386491776], [5.024911165237427, 10.073008686304092, -0.010852077975869179, -0.027709651738405228], [7.908591628074646, 6.30380854010582, 0.03297584503889084, -0.04259300231933594], [3.076299726963043, 5.805655270814896, -0.012030813843011856, -0.07049091905355453], [8.164393901824951, 6.176500171422958, 0.009056603536009789, -0.052246324717998505], [5.058653354644775, 7.600086182355881, 0.020027875900268555, -0.0047928993590176105], [7.2716546058654785, 7.962974309921265, 0.02303464524447918, -0.03739672899246216], [5.602221786975861, 8.390838950872421, -0.018056681379675865, -0.03559279069304466], [7.861390560865402, 7.229218482971191, 0.023161238059401512, -0.04340201988816261], [5.387221723794937, 8.276875466108322, 0.021937236189842224, -0.03687424957752228], [7.11019366979599, 6.8402257561683655, -0.08191091567277908, -0.059565868228673935], [4.757420718669891, 7.447501569986343, 0.010920645669102669, -0.009284867905080318], [7.9335638880729675, 8.475323617458344, -0.011644686572253704, -0.046203333884477615], [5.366178452968597, 5.766912549734116, -0.003529587062075734, -0.018525585532188416], [6.886043697595596, 4.88969162106514, 0.0037842856254428625, -0.034616585820913315], [6.477542817592621, 8.880860656499863, 0.00900083314627409, -0.033021315932273865], [6.302425414323807, 5.8975571393966675, 0.01667255535721779, -0.03794264420866966], [3.64731028676033, 8.524348586797714, -0.058222588151693344, -0.023884959518909454], [5.226862728595734, 3.983318656682968, 0.009688438847661018, -0.024753352627158165], [10.19499734044075, 5.969715267419815, 0.03764183074235916, -0.056404273957014084]]
data = [[239.73072814941406, 27.84111976623535, -0.9995979070663452, 230.37631225585938], [12.322031021118164, 0.9994657039642334, 208.6722412109375, 2.4642207622528076], [0.9996928572654724, 184.638427734375, -0.15480367839336395, 0.9995939135551453], [149.49085998535156, 2.047588348388672, 0.9991946816444397, 115.13711547851562], [9.665715217590332, 0.9996976852416992, 86.158203125, 22.211130142211914], [0.9997919797897339, 47.461875915527344, 48.66405487060547, 0.9998542070388794], [15.213083267211914, 86.24080657958984, 1.000413179397583, 1.5484623908996582], [117.0203628540039, 0.9991615414619446, 0.16019178926944733, 147.54612731933594], [0.999849259853363, 15.805486679077148, 187.88555908203125, 1.0000154972076416], [27.087642669677734, 202.80972290039062, 1.0002251863479614, 48.76532745361328], [218.08692932128906, 0.9998982548713684, 75.46157836914062, 227.11251831054688], [0.999406099319458, 90.4786148071289, 228.5708770751953, 0.9994821548461914], [145.67013549804688, 223.98324584960938, 0.999444305896759, 182.6817626953125], [211.6282501220703, 0.99949711561203, 207.41668701171875, 195.26101684570312], [0.9990322589874268, 230.12014770507812, 173.1173095703125, 0.9991844892501831], [248.07763671875, 147.53407287597656, 0.998986542224884, 254.35841369628906], [127.09947204589844, 0.9990767240524292, 253.7085723876953, 101.0929183959961], [0.9991808533668518, 238.2459716796875, 61.453704833984375, 0.9994721412658691], [222.2607421875, 47.911155700683594, 0.9992184042930603, 208.2094268798828], [39.35700988769531, 0.9999141097068787, 79.13642120361328, 103.60572052001953], [-1.0001789331436157, 101.82093811035156, 96.59516906738281, 0.9989567399024963], [163.76332092285156, 87.16678619384766, -0.9993671774864197, 197.366455078125], [69.30034637451172, 0.9990049004554749, 72.87669372558594, 170.85174560546875], [-1.000073790550232, 105.75251770019531, 177.57362365722656, 0.9998635649681091], [146.85772705078125, 169.6403045654297, 0.9992420673370361, 178.40492248535156], [146.5048065185547, 0.9996330738067627, 207.73294067382812, 122.46027374267578], [0.9995157122612, -0.8888634443283081, -0.8004953861236572, -0.9995357990264893], [-0.9411790370941162, -0.8797743320465088, -0.9995613098144531, -0.9044274091720581], [-0.9362848997116089, -0.9997667670249939, -0.9278189539909363, -0.9331690669059753], [-0.999435544013977, -0.9298252463340759, -0.9232439994812012, -0.9994392395019531], [-0.9442340731620789, -0.9098535180091858, -0.9994693398475647, -0.9332146644592285], [-0.9502681493759155, -0.9997714161872864, -0.9125893712043762, -0.9249767065048218], [-0.9996637105941772, -0.9600564241409302, -0.8926632404327393, -0.9993153214454651], [-0.9468939304351807, -0.9225448369979858, -0.9997684955596924, -0.8988949060440063], [-0.8408340215682983, -0.9996935725212097, -0.9412550330162048, -0.8921312689781189], [-0.9995406270027161, -0.923639178276062, -0.9238823652267456, -0.9997886419296265], [-0.9081968069076538, -0.9190642237663269, -0.9995564818382263, -0.9506633281707764], [-0.944614052772522, -0.999836266040802, -0.9506633281707764, -0.925128698348999], [-0.9997504949569702, -0.9309347867965698, -0.9345217943191528, -0.999657392501831], [-0.9363760948181152, -0.9407990574836731, -0.9996814131736755, -0.9364824891090393], [-0.9342634081840515, -0.999643087387085, -0.9265422224998474, -0.925295889377594], [-0.9997162222862244, -0.9397959113121033, -0.9422429800033569, -0.9996936917304993], [-0.9651481509208679, -0.9185322523117065, -0.9999443292617798, -0.9582781195640564], [-0.9203561544418335, -0.99991375207901, -0.954159140586853, -0.9565150141716003], [-0.9999383091926575, -1.0048636198043823, -0.9842991232872009, -1.0000882148742676], [-0.945328414440155, -0.9644337892532349, -0.999870240688324, -0.9545087218284607], [-0.9611507654190063, -0.9998630285263062, -0.9539007544517517, -0.9610595703125], [-1.0000495910644531, -0.9301292300224304, -0.9468635320663452, -0.9998829364776611]]

plot_drawing(data, output_file="drawing_output.png")